// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "engines/plans/MiopenBatchnormApplicabilityChecks.hpp"

using namespace miopen_legacy_plugin;

namespace
{

// --- Configuration Data Structs ---

struct TensorConfig
{
    int64_t uid;
    std::string name;
    hipdnn_data_sdk::data_objects::DataType dataType;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    std::string description;
    bool isVirtual = false;
    bool isPassByValue = false;
    double passedValue = 0.0;
};

// --- Tensor UID Constants ---

// Common UIDs for all batchnorm operations
struct BnCommonTensorIds
{
    [[maybe_unused]] static constexpr int64_t X = 1;
    [[maybe_unused]] static constexpr int64_t Y = 2;
    [[maybe_unused]] static constexpr int64_t SCALE = 3;
    [[maybe_unused]] static constexpr int64_t BIAS = 4;
    [[maybe_unused]] static constexpr int64_t EPSILON = 5;
    [[maybe_unused]] static constexpr int64_t MEAN = 6;
    [[maybe_unused]] static constexpr int64_t INV_VARIANCE = 7;
};

// Backward-specific
struct BnBackwardTensorIds : BnCommonTensorIds
{
    [[maybe_unused]] static constexpr int64_t DY = 8;
    [[maybe_unused]] static constexpr int64_t DX = 9;
    [[maybe_unused]] static constexpr int64_t DSCALE = 10;
    [[maybe_unused]] static constexpr int64_t DBIAS = 11;
};

// Fused operations (virtual tensors)
struct BnFusedTensorIds : BnBackwardTensorIds
{
    [[maybe_unused]] static constexpr int64_t BN_Y_VIRTUAL = 12;
    [[maybe_unused]] static constexpr int64_t DX_DRELU_VIRTUAL = 13;
};

// Training-specific
struct BnTrainingTensorIds : BnCommonTensorIds
{
    [[maybe_unused]] static constexpr int64_t PREV_RUNNING_MEAN = 8;
    [[maybe_unused]] static constexpr int64_t PREV_RUNNING_VARIANCE = 9;
    [[maybe_unused]] static constexpr int64_t MOMENTUM = 10;
    [[maybe_unused]] static constexpr int64_t NEXT_RUNNING_MEAN = 11;
    [[maybe_unused]] static constexpr int64_t NEXT_RUNNING_VARIANCE = 12;
};

// --- Tensor Role Classification ---
enum class TensorRole
{
    IO, // Input/output tensors (x, y, dx, dy)
    AFFINE, // Scale/bias parameters
    STAT, // Mean/variance statistics
    SCALAR // Pass-by-value scalars (epsilon, momentum)
};

// --- Canonical Tensor Layout Database ---
using hipdnn_data_sdk::utilities::TensorLayout;

namespace canonical_layouts
{

// Test-specific: naming helper (uses TensorLayout.name!)
inline std::string generateName(const std::vector<int64_t>& dims, const TensorLayout& layout)
{
    std::string name = layout.name; // Reuse production name!
    for(auto dim : dims)
    {
        name += "_" + std::to_string(dim);
    }
    return name;
}

// Shape collections (layout-independent)
namespace shapes
{
// 4D shapes for inference/backward (larger spatial dimensions)
inline const std::vector<std::vector<int64_t>> INFERENCE_4D = {
    {1, 3, 56, 56},
    {1, 3, 112, 112},
    {1, 3, 224, 224},
    {2, 3, 224, 224},
    {1, 16, 224, 224},
};

// 4D shapes for training (sufficient spatial: B×S > 1)
inline const std::vector<std::vector<int64_t>> TRAINING_4D = {
    {1, 3, 14, 14},
    {2, 3, 14, 14},
    {1, 3, 28, 28},
};

// 5D shapes for inference/backward
inline const std::vector<std::vector<int64_t>> INFERENCE_5D = {
    {1, 3, 16, 224, 224},
    {1, 4, 16, 224, 224},
    {1, 3, 8, 112, 112},
};

// 5D shapes for training (sufficient spatial: B×S > 1)
inline const std::vector<std::vector<int64_t>> TRAINING_5D = {
    {1, 4, 16, 14, 14},
    {1, 3, 16, 14, 14},
};

// Edge case shapes for specific test scenarios
inline const std::vector<int64_t> DEGENERATE_4D = {1, 1, 1, 1};
inline const std::vector<int64_t> INSUFFICIENT_SPATIAL_4D = {1, 3, 1, 1};
inline const std::vector<int64_t> DIFFERENT_CHANNELS_4D = {1, 5, 224, 224};
} // namespace shapes

// Test-specific: iteration arrays (use struct pointers or references)
inline const std::vector<const TensorLayout*> LAYOUTS_4D
    = {&TensorLayout::NCHW, &TensorLayout::NHWC};

inline const std::vector<const TensorLayout*> LAYOUTS_5D
    = {&TensorLayout::NCDHW, &TensorLayout::NDHWC};

namespace type_configs
{
using DT = hipdnn_data_sdk::data_objects::DataType;

// All invalid type configurations that should be REJECTED
// Documents what type combinations are NOT supported by MIOpen
inline const std::vector<BnTensorTypes> INVALID_ALL = {
    // Invalid IO type (IO must be FLOAT/HALF/BFLOAT16)
    {DT::UINT8, DT::FLOAT, DT::FLOAT}, // IO=UINT8 (unsupported)

    // Invalid affine types (scale/bias must be FLOAT)
    {DT::HALF, DT::HALF, DT::FLOAT}, // IO=HALF, Affine=HALF
    {DT::BFLOAT16, DT::HALF, DT::FLOAT}, // IO=BFLOAT16, Affine=HALF
    {DT::BFLOAT16, DT::BFLOAT16, DT::FLOAT}, // IO=BFLOAT16, Affine=BFLOAT16

    // Invalid stat types (mean/variance must be FLOAT)
    {DT::HALF, DT::FLOAT, DT::HALF}, // IO=HALF, Stat=HALF
    {DT::BFLOAT16, DT::FLOAT, DT::HALF}, // IO=BFLOAT16, Stat=HALF
    {DT::BFLOAT16, DT::FLOAT, DT::BFLOAT16}, // IO=BFLOAT16, Stat=BFLOAT16

    // Both affine and stat invalid (must be FLOAT)
    {DT::HALF, DT::HALF, DT::HALF}, // All HALF
    {DT::BFLOAT16, DT::BFLOAT16, DT::BFLOAT16}, // All BFLOAT16
    {DT::BFLOAT16, DT::HALF, DT::HALF}, // Mixed invalid
};
} // namespace type_configs

} // namespace canonical_layouts

class TensorConfigBuilder
{
public:
    TensorConfigBuilder(int64_t uid, const std::string& name, TensorRole role)
        : _config{uid,
                  name,
                  hipdnn_data_sdk::data_objects::DataType::FLOAT,
                  {},
                  {},
                  "",
                  false,
                  false,
                  0.0}
        , _role(role)
    {
    }

    TensorConfigBuilder& withDataType(hipdnn_data_sdk::data_objects::DataType dt)
    {
        _config.dataType = dt;
        return *this;
    }

    TensorConfigBuilder& withDims(std::vector<int64_t> dims)
    {
        _config.dims = std::move(dims);
        return *this;
    }

    TensorConfigBuilder& withStrides(std::vector<int64_t> strides)
    {
        _config.strides = std::move(strides);
        return *this;
    }

    TensorConfigBuilder& withDescription(const std::string& desc)
    {
        _config.description = desc;
        return *this;
    }

    TensorConfigBuilder& asVirtual()
    {
        _config.isVirtual = true;
        return *this;
    }

    TensorConfigBuilder& asScalar(double value)
    {
        _config.isPassByValue = true;
        _config.passedValue = value;
        _config.dims = {1};
        _config.strides = {1};
        return *this;
    }

    TensorConfig build() const
    {
        return _config;
    }

private:
    TensorConfig _config;
    [[maybe_unused]] TensorRole _role;
};

// --- Factory Helpers ---

inline TensorConfig createIoTensor(int64_t uid,
                                   const std::string& name,
                                   hipdnn_data_sdk::data_objects::DataType dt,
                                   const std::vector<int64_t>& dims,
                                   const std::vector<int64_t>& strides,
                                   bool isVirtual = false)
{
    auto builder = TensorConfigBuilder(uid, name, TensorRole::IO)
                       .withDataType(dt)
                       .withDims(dims)
                       .withStrides(strides);

    if(isVirtual)
    {
        builder.asVirtual();
    }
    return builder.build();
}

[[maybe_unused]] inline TensorConfig createAffineTensor(int64_t uid,
                                                        const std::string& name,
                                                        hipdnn_data_sdk::data_objects::DataType dt,
                                                        const std::vector<int64_t>& derivedDims,
                                                        const std::vector<int64_t>& derivedStrides)
{
    return TensorConfigBuilder(uid, name, TensorRole::AFFINE)
        .withDataType(dt)
        .withDims(derivedDims)
        .withStrides(derivedStrides)
        .build();
}

[[maybe_unused]] inline TensorConfig createStatTensor(int64_t uid,
                                                      const std::string& name,
                                                      hipdnn_data_sdk::data_objects::DataType dt,
                                                      const std::vector<int64_t>& derivedDims,
                                                      const std::vector<int64_t>& derivedStrides)
{
    return TensorConfigBuilder(uid, name, TensorRole::STAT)
        .withDataType(dt)
        .withDims(derivedDims)
        .withStrides(derivedStrides)
        .build();
}

[[maybe_unused]] inline TensorConfig
    createScalarTensor(int64_t uid, const std::string& name, double value)
{
    return TensorConfigBuilder(uid, name, TensorRole::SCALAR).asScalar(value).build();
}

// --- Reusable Tensor Set Factories ---

inline std::vector<TensorConfig> createIoTensorPair(const BnTensorTypes& types,
                                                    const std::vector<int64_t>& dims,
                                                    const TensorLayout& layout,
                                                    int64_t xUid = BnCommonTensorIds::X,
                                                    int64_t yUid = BnCommonTensorIds::Y)
{
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder);
    return {createIoTensor(xUid, "x", types.io, dims, strides),
            createIoTensor(yUid, "y", types.io, dims, strides)};
}

inline std::vector<TensorConfig> createAffineTensorPair(const BnTensorTypes& types,
                                                        const std::vector<int64_t>& baseDims,
                                                        const TensorLayout& layout,
                                                        int64_t scaleUid = BnCommonTensorIds::SCALE,
                                                        int64_t biasUid = BnCommonTensorIds::BIAS)
{
    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(baseDims);
    auto derivedStrides
        = hipdnn_data_sdk::utilities::generateStrides(derivedDims, layout.strideOrder);
    return {createAffineTensor(scaleUid, "scale", types.affine, derivedDims, derivedStrides),
            createAffineTensor(biasUid, "bias", types.affine, derivedDims, derivedStrides)};
}

inline std::vector<TensorConfig> createStatTensorPair(const BnTensorTypes& types,
                                                      const std::vector<int64_t>& baseDims,
                                                      const TensorLayout& layout,
                                                      int64_t meanUid = BnCommonTensorIds::MEAN,
                                                      int64_t invVarianceUid
                                                      = BnCommonTensorIds::INV_VARIANCE)
{
    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(baseDims);
    auto derivedStrides
        = hipdnn_data_sdk::utilities::generateStrides(derivedDims, layout.strideOrder);
    return {
        createStatTensor(meanUid, "mean", types.stat, derivedDims, derivedStrides),
        createStatTensor(invVarianceUid, "inv_variance", types.stat, derivedDims, derivedStrides)};
}

inline std::vector<TensorConfig> createBatchnormInferenceTensors(const BnTensorTypes& types,
                                                                 const std::vector<int64_t>& dims,
                                                                 const TensorLayout& layout)
{
    using UIDs = BnCommonTensorIds;
    std::vector<TensorConfig> configs;

    auto io = createIoTensorPair(types, dims, layout, UIDs::X, UIDs::Y);
    configs.insert(configs.end(), io.begin(), io.end());

    auto affine = createAffineTensorPair(types, dims, layout, UIDs::SCALE, UIDs::BIAS);
    configs.insert(configs.end(), affine.begin(), affine.end());

    auto stat = createStatTensorPair(types, dims, layout, UIDs::MEAN, UIDs::INV_VARIANCE);
    configs.insert(configs.end(), stat.begin(), stat.end());

    return configs;
}

inline std::vector<TensorConfig> createBatchnormTrainingTensors(const BnTensorTypes& types,
                                                                const std::vector<int64_t>& dims,
                                                                const TensorLayout& layout,
                                                                bool includeMeanOutput = false,
                                                                bool includeInvVarianceOutput
                                                                = false)
{
    using UIDs = BnTrainingTensorIds;
    std::vector<TensorConfig> configs;

    auto io = createIoTensorPair(types, dims, layout, UIDs::X, UIDs::Y);
    configs.insert(configs.end(), io.begin(), io.end());

    auto affine = createAffineTensorPair(types, dims, layout, UIDs::SCALE, UIDs::BIAS);
    configs.insert(configs.end(), affine.begin(), affine.end());

    configs.push_back(createScalarTensor(UIDs::EPSILON, "epsilon", 1e-5));

    if(includeMeanOutput || includeInvVarianceOutput)
    {
        auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
        auto derivedStrides
            = hipdnn_data_sdk::utilities::generateStrides(derivedDims, layout.strideOrder);

        if(includeMeanOutput)
        {
            configs.push_back(
                createStatTensor(UIDs::MEAN, "mean", types.stat, derivedDims, derivedStrides));
        }
        if(includeInvVarianceOutput)
        {
            configs.push_back(createStatTensor(
                UIDs::INV_VARIANCE, "inv_variance", types.stat, derivedDims, derivedStrides));
        }
    }

    return configs;
}

inline std::vector<TensorConfig> createBatchnormBackwardTensors(const BnTensorTypes& types,
                                                                const std::vector<int64_t>& dims,
                                                                const TensorLayout& layout,
                                                                bool includeMeanInput = false,
                                                                bool includeInvVarianceInput
                                                                = false)
{
    using UIDs = BnBackwardTensorIds;
    std::vector<TensorConfig> configs;
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder);

    configs.push_back(createIoTensor(UIDs::X, "x", types.io, dims, strides));
    configs.push_back(createIoTensor(UIDs::DY, "dy", types.io, dims, strides));
    configs.push_back(createIoTensor(UIDs::DX, "dx", types.io, dims, strides));

    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides
        = hipdnn_data_sdk::utilities::generateStrides(derivedDims, layout.strideOrder);
    configs.push_back(
        createAffineTensor(UIDs::SCALE, "scale", types.affine, derivedDims, derivedStrides));
    configs.push_back(
        createAffineTensor(UIDs::DSCALE, "dscale", types.affine, derivedDims, derivedStrides));
    configs.push_back(
        createAffineTensor(UIDs::DBIAS, "dbias", types.affine, derivedDims, derivedStrides));

    if(includeMeanInput)
    {
        configs.push_back(
            createStatTensor(UIDs::MEAN, "mean", types.stat, derivedDims, derivedStrides));
    }
    if(includeInvVarianceInput)
    {
        configs.push_back(createStatTensor(
            UIDs::INV_VARIANCE, "inv_variance", types.stat, derivedDims, derivedStrides));
    }

    return configs;
}

inline std::vector<TensorConfig> createBatchnormFusedBackwardTensors(
    const BnTensorTypes& types, const std::vector<int64_t>& dims, const TensorLayout& layout)
{
    using UIDs = BnFusedTensorIds;
    std::vector<TensorConfig> configs;
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder);
    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides
        = hipdnn_data_sdk::utilities::generateStrides(derivedDims, layout.strideOrder);

    // Forward inputs (x, scale, bias, mean, invVariance)
    configs.push_back(createIoTensor(UIDs::X, "x", types.io, dims, strides));
    configs.push_back(
        createAffineTensor(UIDs::SCALE, "scale", types.affine, derivedDims, derivedStrides));
    configs.push_back(
        createAffineTensor(UIDs::BIAS, "bias", types.affine, derivedDims, derivedStrides));
    configs.push_back(
        createStatTensor(UIDs::MEAN, "mean", types.stat, derivedDims, derivedStrides));
    configs.push_back(createStatTensor(
        UIDs::INV_VARIANCE, "inv_variance", types.stat, derivedDims, derivedStrides));

    // Backward inputs (dy)
    configs.push_back(createIoTensor(UIDs::DY, "dy", types.io, dims, strides));

    // Backward outputs (dx, dscale, dbias)
    configs.push_back(createIoTensor(UIDs::DX, "dx", types.io, dims, strides));
    configs.push_back(
        createAffineTensor(UIDs::DSCALE, "dscale", types.affine, derivedDims, derivedStrides));
    configs.push_back(
        createAffineTensor(UIDs::DBIAS, "dbias", types.affine, derivedDims, derivedStrides));

    // Virtual tensors (BN_Y, DX_drelu)
    configs.push_back(createIoTensor(UIDs::BN_Y_VIRTUAL, "BN_Y", types.io, dims, strides, true));
    configs.push_back(
        createIoTensor(UIDs::DX_DRELU_VIRTUAL, "DX_drelu", types.io, dims, strides, true));

    return configs;
}

// --- BnTensorTypes Helpers ---

inline std::string dataTypeToString(hipdnn_data_sdk::data_objects::DataType dt)
{
    using DT = hipdnn_data_sdk::data_objects::DataType;
    switch(dt)
    {
    case DT::FLOAT:
        return "Float";
    case DT::HALF:
        return "Half";
    case DT::BFLOAT16:
        return "Bfloat16";
    default:
        return "Unknown";
    }
}

inline std::string toString(const BnTensorTypes& types)
{
    return "IO_" + dataTypeToString(types.io) + "_Affine_" + dataTypeToString(types.affine)
           + "_Stat_" + dataTypeToString(types.stat);
}

inline std::string activationModeToString(hipdnn_data_sdk::data_objects::PointwiseMode mode)
{
    using PM = hipdnn_data_sdk::data_objects::PointwiseMode;
    switch(mode)
    {
    case PM::IDENTITY:
        return "Identity";
    case PM::RELU_BWD:
        return "ReluBwd";
    case PM::SIGMOID_BWD:
        return "SigmoidBwd";
    case PM::TANH_BWD:
        return "TanhBwd";
    case PM::RELU_FWD:
        return "ReluFwd";
    default:
        return "Unknown";
    }
}

// --- Test Case Structs: Layer 1 (Atomic Validators) ---

struct DimensionCountTestCase
{
    std::string name;
    bool shouldPass;
    size_t numDims;

    friend std::ostream& operator<<(std::ostream& os, const DimensionCountTestCase& tc)
    {
        return os << tc.name;
    }
};

struct SupportedLayoutTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<int64_t> strideOrder;
    size_t numDims;

    friend std::ostream& operator<<(std::ostream& os, const SupportedLayoutTestCase& tc)
    {
        return os << tc.name;
    }
};

struct TensorDescriptorListTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<std::vector<int64_t>> tensorDims;
    std::vector<std::vector<int64_t>> tensorStrides;

    friend std::ostream& operator<<(std::ostream& os, const TensorDescriptorListTestCase& tc)
    {
        return os << tc.name;
    }
};

struct DataTypeIsSupportedTestCase
{
    std::string name;
    bool shouldPass;
    hipdnn_data_sdk::data_objects::DataType dataType;
    std::unordered_set<hipdnn_data_sdk::data_objects::DataType> allowedTypes;

    friend std::ostream& operator<<(std::ostream& os, const DataTypeIsSupportedTestCase& tc)
    {
        return os << tc.name;
    }
};

struct ConsistentDataTypesTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    std::vector<int64_t> tensorIds;
    std::unordered_set<hipdnn_data_sdk::data_objects::DataType> allowedTypes;

    friend std::ostream& operator<<(std::ostream& os, const ConsistentDataTypesTestCase& tc)
    {
        return os << tc.name;
    }
};

struct FixedDataTypeTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    std::vector<int64_t> tensorIds;
    hipdnn_data_sdk::data_objects::DataType expectedType;

    friend std::ostream& operator<<(std::ostream& os, const FixedDataTypeTestCase& tc)
    {
        return os << tc.name;
    }
};

struct ConsistentShapesTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    std::vector<int64_t> tensorIds;
    std::vector<int64_t> referenceShape;

    friend std::ostream& operator<<(std::ostream& os, const ConsistentShapesTestCase& tc)
    {
        return os << tc.name;
    }
};

struct SpatialDimensionsTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<int64_t> ioDims;

    friend std::ostream& operator<<(std::ostream& os, const SpatialDimensionsTestCase& tc)
    {
        return os << tc.name;
    }
};

// --- Test Case Structs: Layer 2 (Component Validators) ---

struct TensorLayoutsAndDimsTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;

    friend std::ostream& operator<<(std::ostream& os, const TensorLayoutsAndDimsTestCase& tc)
    {
        return os << tc.name;
    }
};

struct TensorDataTypesComponentTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    std::vector<int64_t> ioTensorIds;
    std::vector<int64_t> affineTensorIds;
    std::vector<int64_t> statTensorIds;

    friend std::ostream& operator<<(std::ostream& os, const TensorDataTypesComponentTestCase& tc)
    {
        return os << tc.name;
    }
};

struct TensorShapesComponentTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    std::vector<int64_t> ioTensorIds;
    std::vector<int64_t> affineTensorIds;
    std::vector<int64_t> statTensorIds;
    bool isTraining;

    friend std::ostream& operator<<(std::ostream& os, const TensorShapesComponentTestCase& tc)
    {
        return os << tc.name;
    }
};

// --- Test Case Structs: Layer 3 (High-Level Validators) ---

struct BatchnormInferenceConfigTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    int64_t xUid;
    int64_t yUid;
    int64_t scaleUid;
    int64_t biasUid;
    int64_t meanUid;
    int64_t invVarianceUid;

    friend std::ostream& operator<<(std::ostream& os, const BatchnormInferenceConfigTestCase& tc)
    {
        return os << tc.name;
    }
};

struct BatchnormTrainingConfigTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    int64_t xUid;
    int64_t yUid;
    int64_t scaleUid;
    int64_t biasUid;
    int64_t epsilonUid;
    flatbuffers::Optional<int64_t> meanUid;
    flatbuffers::Optional<int64_t> invVarianceUid;

    friend std::ostream& operator<<(std::ostream& os, const BatchnormTrainingConfigTestCase& tc)
    {
        return os << tc.name;
    }
};

struct BatchnormBackwardConfigTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    int64_t xUid;
    int64_t dyUid;
    int64_t dxUid;
    int64_t scaleUid;
    int64_t dscaleUid;
    int64_t dbiasUid;
    flatbuffers::Optional<int64_t> meanUid;
    flatbuffers::Optional<int64_t> invVarianceUid;

    friend std::ostream& operator<<(std::ostream& os, const BatchnormBackwardConfigTestCase& tc)
    {
        return os << tc.name;
    }
};

struct BatchnormFusedBackwardConfigTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    int64_t xUid;
    int64_t scaleUid;
    int64_t biasUid;
    int64_t meanUid;
    int64_t invVarianceUid;
    int64_t dyUid;
    int64_t dxUid;
    int64_t dscaleUid;
    int64_t dbiasUid;
    int64_t bnYVirtualUid;
    int64_t dxDreluVirtualUid;
    hipdnn_data_sdk::data_objects::PointwiseMode activationMode;

    friend std::ostream& operator<<(std::ostream& os,
                                    const BatchnormFusedBackwardConfigTestCase& tc)
    {
        return os << tc.name;
    }
};

struct ActivationModeTestCase
{
    std::string name;
    bool shouldPass;
    hipdnn_data_sdk::data_objects::PointwiseMode mode;
    flatbuffers::Optional<double> reluLowerClip;
    flatbuffers::Optional<double> reluUpperClip;
    flatbuffers::Optional<double> reluLowerClipSlope;

    friend std::ostream& operator<<(std::ostream& os, const ActivationModeTestCase& tc)
    {
        return os << tc.name;
    }
};

// --- Helper Functions ---

std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*> buildTensorMap(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>&
        tensorOffsets)
{
    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test_graph",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorOffsets,
        nullptr);

    builder.Finish(graphOffset);

    const auto* graph = hipdnn_data_sdk::data_objects::GetGraph(builder.GetBufferPointer());
    std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*> tensorMap;

    if(graph->tensors() != nullptr)
    {
        for(const auto* tensorAttr : *graph->tensors())
        {
            tensorMap[tensorAttr->uid()] = tensorAttr;
        }
    }

    return tensorMap;
}

auto buildTensorMapFromConfigs(const std::vector<TensorConfig>& configs)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>> tensorOffsets;
    tensorOffsets.reserve(configs.size());

    for(const auto& config : configs)
    {
        if(config.isPassByValue)
        {
            // Create a pass-by-value tensor with embedded scalar value
            hipdnn_data_sdk::data_objects::Float64Value floatValue(config.passedValue);
            auto valueOffset = builder.CreateStruct(floatValue).Union();
            tensorOffsets.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                config.uid,
                config.name.c_str(),
                config.dataType,
                &config.strides,
                &config.dims,
                config.isVirtual,
                hipdnn_data_sdk::data_objects::TensorValue::Float64Value,
                valueOffset));
        }
        else
        {
            tensorOffsets.push_back(
                hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                            config.uid,
                                                                            config.name.c_str(),
                                                                            config.dataType,
                                                                            &config.strides,
                                                                            &config.dims,
                                                                            config.isVirtual));
        }
    }

    auto tensorMap = buildTensorMap(builder, tensorOffsets);
    return std::make_pair(std::move(builder), std::move(tensorMap));
}

// --- Graph Builders ---

inline flatbuffers::FlatBufferBuilder
    buildBatchnormInferenceGraph(const std::vector<TensorConfig>& configs,
                                 int64_t xUid,
                                 int64_t yUid,
                                 int64_t scaleUid,
                                 int64_t biasUid,
                                 int64_t meanUid,
                                 int64_t invVarianceUid)
{
    flatbuffers::FlatBufferBuilder builder;

    // Build tensor attribute offsets from configs
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>> tensorOffsets;
    tensorOffsets.reserve(configs.size());
    for(const auto& cfg : configs)
    {
        tensorOffsets.push_back(
            hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                        cfg.uid,
                                                                        cfg.name.c_str(),
                                                                        cfg.dataType,
                                                                        &cfg.strides,
                                                                        &cfg.dims,
                                                                        cfg.isVirtual));
    }

    // Create BatchnormInferenceAttributes with specified UIDs
    auto bnInfAttrs = hipdnn_data_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, xUid, meanUid, invVarianceUid, scaleUid, biasUid, yUid);

    // Create node
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_inference",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttrs.Union()));

    // Build graph
    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test_graph",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorOffsets,
        &nodes);

    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    buildBatchnormTrainingGraph(const std::vector<TensorConfig>& configs,
                                int64_t xUid,
                                int64_t yUid,
                                int64_t scaleUid,
                                int64_t biasUid,
                                int64_t epsilonUid,
                                flatbuffers::Optional<int64_t> meanUid = flatbuffers::nullopt,
                                flatbuffers::Optional<int64_t> invVarianceUid
                                = flatbuffers::nullopt)
{
    flatbuffers::FlatBufferBuilder builder;

    // Build tensor attribute offsets
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>> tensorOffsets;
    tensorOffsets.reserve(configs.size());
    for(const auto& cfg : configs)
    {
        if(cfg.isPassByValue)
        {
            // Create a pass-by-value tensor with embedded scalar value
            hipdnn_data_sdk::data_objects::Float64Value floatValue(cfg.passedValue);
            auto valueOffset = builder.CreateStruct(floatValue).Union();
            tensorOffsets.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                cfg.uid,
                cfg.name.c_str(),
                cfg.dataType,
                &cfg.strides,
                &cfg.dims,
                cfg.isVirtual,
                hipdnn_data_sdk::data_objects::TensorValue::Float64Value,
                valueOffset));
        }
        else
        {
            tensorOffsets.push_back(
                hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                            cfg.uid,
                                                                            cfg.name.c_str(),
                                                                            cfg.dataType,
                                                                            &cfg.strides,
                                                                            &cfg.dims,
                                                                            cfg.isVirtual));
        }
    }

    // Create BatchnormAttributes (training mode) with specified UIDs
    auto bnAttrs = hipdnn_data_sdk::data_objects::CreateBatchnormAttributes(
        builder,
        xUid,
        scaleUid,
        biasUid,
        epsilonUid,
        0, // peer_stats_tensor_uid
        flatbuffers::nullopt, // prev_running_mean_tensor_uid
        flatbuffers::nullopt, // prev_running_variance_tensor_uid
        flatbuffers::nullopt, // momentum_tensor_uid
        yUid,
        meanUid,
        invVarianceUid,
        flatbuffers::nullopt, // next_running_mean_tensor_uid
        flatbuffers::nullopt // next_running_variance_tensor_uid
    );

    // Create node
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_training",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnAttrs.Union()));

    // Build graph
    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test_graph",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorOffsets,
        &nodes);

    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    buildBatchnormBackwardGraph(const std::vector<TensorConfig>& configs,
                                int64_t xUid,
                                int64_t dyUid,
                                int64_t dxUid,
                                int64_t scaleUid,
                                int64_t dscaleUid,
                                int64_t dbiasUid,
                                flatbuffers::Optional<int64_t> meanUid = flatbuffers::nullopt,
                                flatbuffers::Optional<int64_t> invVarianceUid
                                = flatbuffers::nullopt)
{
    flatbuffers::FlatBufferBuilder builder;

    // Build tensor attribute offsets
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>> tensorOffsets;
    tensorOffsets.reserve(configs.size());
    for(const auto& cfg : configs)
    {
        tensorOffsets.push_back(
            hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                        cfg.uid,
                                                                        cfg.name.c_str(),
                                                                        cfg.dataType,
                                                                        &cfg.strides,
                                                                        &cfg.dims,
                                                                        cfg.isVirtual));
    }

    // Create BatchnormBackwardAttributes with specified UIDs
    auto bnBwdAttrs = hipdnn_data_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        dyUid,
        xUid,
        meanUid,
        invVarianceUid,
        scaleUid,
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(), // peer_stats_tensor_uid
        dxUid,
        dscaleUid,
        dbiasUid);

    // Create node
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_backward",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttrs.Union()));

    // Build graph
    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test_graph",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorOffsets,
        &nodes);

    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    buildBatchnormFusedBackwardGraph(const std::vector<TensorConfig>& configs,
                                     int64_t xUid,
                                     int64_t scaleUid,
                                     int64_t biasUid,
                                     int64_t meanUid,
                                     int64_t invVarianceUid,
                                     int64_t dyUid,
                                     int64_t dxUid,
                                     int64_t dscaleUid,
                                     int64_t dbiasUid,
                                     int64_t bnYVirtualUid,
                                     int64_t dxDreluVirtualUid,
                                     hipdnn_data_sdk::data_objects::PointwiseMode activationMode
                                     = hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD)
{
    flatbuffers::FlatBufferBuilder builder;

    // Build tensor attribute offsets
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>> tensorOffsets;
    tensorOffsets.reserve(configs.size());
    for(const auto& cfg : configs)
    {
        tensorOffsets.push_back(
            hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                        cfg.uid,
                                                                        cfg.name.c_str(),
                                                                        cfg.dataType,
                                                                        &cfg.strides,
                                                                        &cfg.dims,
                                                                        cfg.isVirtual));
    }

    // Create 3 nodes for the fusion
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;

    // Node 0: Batchnorm Inference
    auto bnInfAttrs = hipdnn_data_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, xUid, meanUid, invVarianceUid, scaleUid, biasUid, bnYVirtualUid);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_inference",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttrs.Union()));

    // Node 1: Activation (Backward mode, e.g., RELU_BWD)
    auto actAttrs = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        activationMode,
        flatbuffers::nullopt, // relu_lower_clip
        flatbuffers::nullopt, // relu_upper_clip
        flatbuffers::nullopt, // relu_lower_clip_slope
        flatbuffers::nullopt, // axis_tensor_uid
        bnYVirtualUid,
        flatbuffers::Optional<int64_t>(dyUid),
        flatbuffers::nullopt, // in_2_tensor_uid
        dxDreluVirtualUid);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "activation_bwd",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttrs.Union()));

    // Node 2: Batchnorm Backward
    auto bnBwdAttrs = hipdnn_data_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        dxDreluVirtualUid,
        xUid,
        flatbuffers::Optional<int64_t>(meanUid),
        flatbuffers::Optional<int64_t>(invVarianceUid),
        scaleUid,
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(), // peer_stats_tensor_uid
        dxUid,
        dscaleUid,
        dbiasUid);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_backward",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttrs.Union()));

    // Build graph
    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test_graph",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorOffsets,
        &nodes);

    builder.Finish(graphOffset);
    return builder;
}

// --- Test Data Providers: Layer 1 (Atomic Validators) ---

// Only 4D and 5D tensors work with batchnorm
inline std::vector<DimensionCountTestCase> getValidateDimensionCountTestCases()
{
    return {
        // Happy paths - supported dimensions
        {"Accepts4D", true, 4},
        {"Accepts5D", true, 5},

        // Unhappy paths - unsupported dimensions
        {"Rejects3D", false, 3},
        {"Rejects6D", false, 6},
        {"Rejects2D", false, 2},
        {"Rejects1D", false, 1},
    };
}

// Only NCHW/NHWC (4D) and NCDHW/NDHWC (5D) layouts are supported
inline std::vector<SupportedLayoutTestCase> getValidateSupportedLayoutTestCases()
{
    return {
        // Happy paths - 4D supported layouts
        {"AcceptsNchw4D", true, {3, 2, 1, 0}, 4}, // NCHW stride order
        {"AcceptsNhwc4D", true, {3, 0, 2, 1}, 4}, // NHWC stride order

        // Happy paths - 5D supported layouts
        {"AcceptsNcdhw5D", true, {4, 3, 2, 1, 0}, 5}, // NCDHW stride order
        {"AcceptsNdhwc5D", true, {4, 0, 3, 2, 1}, 5}, // NDHWC stride order

        // Unhappy paths - unsupported 4D layouts
        {"RejectsInvalid4D_AllReversed", false, {0, 1, 2, 3}, 4},
        {"RejectsInvalid4D_Random", false, {2, 1, 0, 3}, 4},

        // Unhappy paths - unsupported 5D layouts
        {"RejectsInvalid5D_AllReversed", false, {0, 1, 2, 3, 4}, 5},
        {"RejectsInvalid5D_Random", false, {1, 2, 3, 4, 0}, 5},
    };
}

// All tensors must have the same number of dimensions
inline std::vector<TensorDescriptorListTestCase> getValidateConsistentDimensionsTestCases()
{
    using namespace canonical_layouts;

    auto dims4D = shapes::INFERENCE_4D[2]; // {1, 3, 224, 224}
    auto dims5D = shapes::INFERENCE_5D[0]; // {1, 3, 16, 224, 224}
    auto strides4D
        = hipdnn_data_sdk::utilities::generateStrides(dims4D, TensorLayout::NCHW.strideOrder);
    auto strides5D
        = hipdnn_data_sdk::utilities::generateStrides(dims5D, TensorLayout::NCDHW.strideOrder);

    return {
        // Happy paths - consistent dimensions
        {"AcceptsSame4D_TwoTensors", true, {dims4D, dims4D}, {strides4D, strides4D}},
        {"AcceptsSame5D_ThreeTensors",
         true,
         {dims5D, dims5D, dims5D},
         {strides5D, strides5D, strides5D}},
        {"AcceptsEmpty", true, {}, {}},
        {"AcceptsSingleTensor", true, {dims4D}, {strides4D}},

        // Unhappy paths - inconsistent dimensions
        {"RejectsMixed4D5D", false, {dims4D, dims5D}, {strides4D, strides5D}},
        {"RejectsMixed5D4D", false, {dims5D, dims4D}, {strides5D, strides4D}},
    };
}

// All tensors must be packed (contiguous in memory)
inline std::vector<TensorDescriptorListTestCase> getValidatePackedTensorsTestCases()
{
    using namespace canonical_layouts;

    auto dims4D = shapes::INFERENCE_4D[2]; // {1, 3, 224, 224}
    auto dims5D = shapes::INFERENCE_5D[0]; // {1, 3, 16, 224, 224}
    auto nchwStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims4D, TensorLayout::NCHW.strideOrder);
    auto nhwcStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims4D, TensorLayout::NHWC.strideOrder);
    auto ncdhwStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims5D, TensorLayout::NCDHW.strideOrder);

    return {
        // Happy paths - packed tensors
        {"AcceptsPacked4D_NCHW", true, {dims4D}, {nchwStrides}},
        {"AcceptsPacked4D_NHWC", true, {dims4D}, {nhwcStrides}},
        {"AcceptsPacked5D_NCDHW", true, {dims5D}, {ncdhwStrides}},
        {"AcceptsMultiplePacked", true, {dims4D, dims4D}, {nchwStrides, nhwcStrides}},

        // Unhappy paths - non-packed tensors
        {"RejectsNonPacked_ExtraStride", false, {dims4D}, {{200000, 60000, 250, 1}}},
        {"RejectsNonPacked_Gaps", false, {dims4D}, {{151000, 50200, 225, 1}}},
        {"RejectsOneNonPacked", false, {dims4D, dims4D}, {nchwStrides, {200000, 60000, 250, 1}}},
    };
}

// All tensors must have the same layout
inline std::vector<TensorDescriptorListTestCase> getValidateConsistentLayoutsTestCases()
{
    using namespace canonical_layouts;

    auto dims4D = shapes::INFERENCE_4D[2]; // {1, 3, 224, 224}
    auto dims5D = shapes::INFERENCE_5D[0]; // {1, 3, 16, 224, 224}
    auto nchwStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims4D, TensorLayout::NCHW.strideOrder);
    auto nhwcStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims4D, TensorLayout::NHWC.strideOrder);
    auto ncdhwStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims5D, TensorLayout::NCDHW.strideOrder);
    auto ndhwcStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims5D, TensorLayout::NDHWC.strideOrder);
    auto degenerateStrides = hipdnn_data_sdk::utilities::generateStrides(
        shapes::DEGENERATE_4D, TensorLayout::NCHW.strideOrder);

    return {
        // Happy paths - consistent layouts
        {"AcceptsSameNchw", true, {dims4D, dims4D}, {nchwStrides, nchwStrides}},
        {"AcceptsSameNhwc", true, {dims4D, dims4D}, {nhwcStrides, nhwcStrides}},
        {"AcceptsSameNcdhw", true, {dims5D, dims5D}, {ncdhwStrides, ncdhwStrides}},
        {"AcceptsWithDegenerate",
         true, // Degenerate tensors (all dims=1) are layout-agnostic
         {dims4D, shapes::DEGENERATE_4D},
         {nchwStrides, degenerateStrides}},

        // Unhappy paths - inconsistent layouts
        {"RejectsMixedNchwNhwc", false, {dims4D, dims4D}, {nchwStrides, nhwcStrides}},
        {"RejectsMixedNcdhwNdhwc", false, {dims5D, dims5D}, {ncdhwStrides, ndhwcStrides}},
    };
}

inline std::vector<DataTypeIsSupportedTestCase> getValidateDataTypeIsSupportedTestCases()
{
    using DT = hipdnn_data_sdk::data_objects::DataType;
    std::unordered_set<DT> ioTypes = {DT::FLOAT, DT::HALF, DT::BFLOAT16};

    return {
        // Happy paths - supported types
        {"AcceptsFloat", true, DT::FLOAT, ioTypes},
        {"AcceptsHalf", true, DT::HALF, ioTypes},
        {"AcceptsBfloat16", true, DT::BFLOAT16, ioTypes},

        // Unhappy paths - unsupported types
        {"RejectsUint8", false, DT::UINT8, ioTypes},
        {"RejectsWithEmptyAllowedList", false, DT::FLOAT, {}}, // Edge case
    };
}

// Training requires B × S > 1 (batch × spatial product)
inline std::vector<SpatialDimensionsTestCase> getValidateSpatialDimensionsTestCases()
{
    using namespace canonical_layouts;

    return {
        // Happy paths - sufficient spatial dimensions (B × S > 1)
        {"AcceptsSufficientSpatial4D_LargeBatch",
         true,
         shapes::INFERENCE_4D[3]}, // {2, 3, 224, 224}
        {"AcceptsSufficientSpatial5D", true, shapes::INFERENCE_5D[0]}, // {1, 3, 16, 224, 224}
        {"AcceptsBatch2Spatial1", true, {2, 3, 1, 1}}, // B=2, S=1 → B×S=2 > 1
        {"AcceptsBatch1Spatial2", true, {1, 3, 2, 1}}, // B=1, S=2 → B×S=2 > 1
        {"AcceptsLargeSpatial", true, {1, 3, 512, 512}},

        // Unhappy paths - insufficient spatial dimensions (B × S ≤ 1)
        {"RejectsBatch1Spatial1_4D", false, shapes::INSUFFICIENT_SPATIAL_4D}, // {1, 3, 1, 1}
        {"RejectsBatch1Spatial1_5D", false, {1, 3, 1, 1, 1}}, // B=1, S=1 → B×S=1 ≤ 1
        {"RejectsZeroBatch", false, {0, 3, 224, 224}}, // B=0 → B×S=0 ≤ 1
        {"RejectsZeroSpatial", false, {2, 3, 0, 0}}, // S=0 → B×S=0 ≤ 1
    };
}

// Validates a group of tensors all have the same data type from allowed list
inline std::vector<ConsistentDataTypesTestCase> getValidateConsistentDataTypesTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;

    const auto& testDims = shapes::INFERENCE_4D[2]; // {1, 3, 224, 224}
    auto testStrides
        = hipdnn_data_sdk::utilities::generateStrides(testDims, TensorLayout::NCHW.strideOrder);

    return {
        // Happy paths - consistent types
        {"AcceptsConsistentFloat",
         true,
         {createIoTensor(1, "float_tensor_a", DT::FLOAT, testDims, testStrides),
          createIoTensor(2, "float_tensor_b", DT::FLOAT, testDims, testStrides)},
         {1, 2},
         bn_type_configs::getAllowedIoTypes()},
        {"AcceptsConsistentHalf",
         true,
         {createIoTensor(1, "half_tensor_a", DT::HALF, testDims, testStrides),
          createIoTensor(2, "half_tensor_b", DT::HALF, testDims, testStrides)},
         {1, 2},
         bn_type_configs::getAllowedIoTypes()},
        {"AcceptsSingleTensor",
         true,
         {createIoTensor(1, "single_tensor", DT::FLOAT, testDims, testStrides)},
         {1},
         bn_type_configs::getAllowedIoTypes()},
        {"AcceptsEmptyTensorList", true, {}, {}, bn_type_configs::getAllowedIoTypes()},

        // Unhappy paths - inconsistent types or unsupported types
        {"RejectsInconsistentTypes_FloatHalf",
         false,
         {createIoTensor(1, "float_tensor", DT::FLOAT, testDims, testStrides),
          createIoTensor(2, "half_tensor", DT::HALF, testDims, testStrides)},
         {1, 2},
         bn_type_configs::getAllowedIoTypes()},
        {"RejectsUnsupportedType",
         false,
         {createIoTensor(1, "unsupported_uint8_tensor", DT::UINT8, testDims, testStrides)},
         {1},
         bn_type_configs::getAllowedIoTypes()},
    };
}

// Validates all specified tensors have a specific required data type
inline std::vector<FixedDataTypeTestCase> getValidateFixedDataTypeTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;

    auto derivedDims
        = hipdnn_data_sdk::utilities::getDerivedShape(shapes::INFERENCE_4D[2]); // {1, 3, 1, 1}
    auto derivedStrides
        = hipdnn_data_sdk::utilities::generateStrides(derivedDims, TensorLayout::NCHW.strideOrder);

    return {
        // Happy paths - matching types
        {"AcceptsMatchingFloat",
         true,
         {{1, "t1", DT::FLOAT, derivedDims, derivedStrides, ""}},
         {1},
         DT::FLOAT},
        {"AcceptsMultipleMatchingFloat",
         true,
         {{1, "t1", DT::FLOAT, derivedDims, derivedStrides, ""},
          {2, "t2", DT::FLOAT, derivedDims, derivedStrides, ""}},
         {1, 2},
         DT::FLOAT},
        {"AcceptsMatchingHalf",
         true,
         {{1, "t1", DT::HALF, derivedDims, derivedStrides, ""}},
         {1},
         DT::HALF},

        // Unhappy paths - mismatched types
        {"RejectsMismatchedType_HalfExpectFloat",
         false,
         {{1, "t1", DT::HALF, derivedDims, derivedStrides, ""}},
         {1},
         DT::FLOAT},
        {"RejectsMismatchedType_FloatExpectHalf",
         false,
         {{1, "t1", DT::FLOAT, derivedDims, derivedStrides, ""}},
         {1},
         DT::HALF},
        {"RejectsOneMismatch",
         false,
         {{1, "t1", DT::FLOAT, derivedDims, derivedStrides, ""},
          {2, "t2", DT::HALF, derivedDims, derivedStrides, ""}},
         {1, 2},
         DT::FLOAT},
    };
}

// Validates all tensors have the same shape
inline std::vector<ConsistentShapesTestCase> getValidateConsistentShapesTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;

    auto canonicalDims = shapes::INFERENCE_4D[2]; // {1, 3, 224, 224}
    auto canonicalStrides = hipdnn_data_sdk::utilities::generateStrides(
        canonicalDims, TensorLayout::NCHW.strideOrder);
    auto mediumDims = shapes::INFERENCE_4D[1]; // {1, 3, 112, 112}
    auto mediumStrides
        = hipdnn_data_sdk::utilities::generateStrides(mediumDims, TensorLayout::NCHW.strideOrder);
    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(canonicalDims); // {1, 3, 1, 1}
    auto derivedStrides
        = hipdnn_data_sdk::utilities::generateStrides(derivedDims, TensorLayout::NCHW.strideOrder);
    auto differentChannelsDims = shapes::DIFFERENT_CHANNELS_4D; // {1, 5, 224, 224}
    auto differentChannelsStrides = hipdnn_data_sdk::utilities::generateStrides(
        differentChannelsDims, TensorLayout::NCHW.strideOrder);

    return {
        // Happy paths - matching shapes
        {"AcceptsMatchingShape",
         true,
         {{1, "t1", DT::FLOAT, canonicalDims, canonicalStrides, ""}},
         {1},
         canonicalDims},
        {"AcceptsMultipleMatchingShapes",
         true,
         {{1, "t1", DT::FLOAT, canonicalDims, canonicalStrides, ""},
          {2, "t2", DT::FLOAT, canonicalDims, canonicalStrides, ""}},
         {1, 2},
         canonicalDims},
        {"AcceptsDerivedShape",
         true,
         {{1, "t1", DT::FLOAT, derivedDims, derivedStrides, ""}},
         {1},
         derivedDims},

        // Unhappy paths - mismatched shapes
        {"RejectsMismatchedShape_DifferentSpatial",
         false,
         {{1, "t1", DT::FLOAT, mediumDims, mediumStrides, ""}},
         {1},
         canonicalDims},
        {"RejectsMismatchedShape_DifferentChannels",
         false,
         {{1, "t1", DT::FLOAT, differentChannelsDims, differentChannelsStrides, ""}},
         {1},
         canonicalDims},
        {"RejectsOneMismatch",
         false,
         {{1, "t1", DT::FLOAT, canonicalDims, canonicalStrides, ""},
          {2, "t2", DT::FLOAT, mediumDims, mediumStrides, ""}},
         {1, 2},
         canonicalDims},
    };
}

// --- Test Data Providers: Layer 2 (Component Validators) ---

inline std::vector<TensorLayoutsAndDimsTestCase> getCheckTensorLayoutsAndDimsSupportedTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;

    auto dims4D = shapes::INFERENCE_4D[2]; // {1, 3, 224, 224}
    auto dims5D = shapes::INFERENCE_5D[0]; // {1, 3, 16, 224, 224}
    auto nchwStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims4D, TensorLayout::NCHW.strideOrder);
    auto nhwcStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims4D, TensorLayout::NHWC.strideOrder);
    auto ncdhwStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims5D, TensorLayout::NCDHW.strideOrder);
    auto ndhwcStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims5D, TensorLayout::NDHWC.strideOrder);

    return {
        // Happy paths - valid layouts and dimensions
        {"AcceptsNchw4D", true, {{1, "x", DT::FLOAT, dims4D, nchwStrides, ""}}},
        {"AcceptsNhwc4D", true, {{1, "x", DT::FLOAT, dims4D, nhwcStrides, ""}}},
        {"AcceptsNcdhw5D", true, {{1, "x", DT::FLOAT, dims5D, ncdhwStrides, ""}}},
        {"AcceptsNdhwc5D", true, {{1, "x", DT::FLOAT, dims5D, ndhwcStrides, ""}}},

        // Unhappy paths - mixed layouts
        {"RejectsMixedNchwNhwc",
         false,
         {{1, "x1", DT::FLOAT, dims4D, nchwStrides, ""},
          {2, "x2", DT::FLOAT, dims4D, nhwcStrides, ""}}},

        // Unhappy paths - mixed dimensions
        {"RejectsMixed4D5D",
         false,
         {{1, "x1", DT::FLOAT, dims4D, nchwStrides, ""},
          {2, "x2", DT::FLOAT, dims5D, ncdhwStrides, ""}}},

        // Unhappy paths - non-packed tensors
        {"RejectsNonPacked", false, {{1, "x", DT::FLOAT, dims4D, {200000, 60000, 250, 1}, ""}}},
    };
}

// IO: FLOAT/HALF/BFLOAT16, Affine: FLOAT, Stat: FLOAT
inline std::vector<TensorDataTypesComponentTestCase> getCheckTensorDataTypesSupportedTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;

    auto ioDims = shapes::INFERENCE_4D[2]; // {1, 3, 224, 224}
    auto ioStrides
        = hipdnn_data_sdk::utilities::generateStrides(ioDims, TensorLayout::NCHW.strideOrder);
    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(ioDims); // {1, 3, 1, 1}
    auto derivedStrides
        = hipdnn_data_sdk::utilities::generateStrides(derivedDims, TensorLayout::NCHW.strideOrder);

    return {
        // Happy paths - valid IO data types (FLOAT, HALF, BFLOAT16) with FLOAT affine/stat
        {"AcceptsFloat_IO",
         true,
         {{1, "x", DT::FLOAT, ioDims, ioStrides, ""},
          {2, "y", DT::FLOAT, ioDims, ioStrides, ""},
          {3, "scale", DT::FLOAT, derivedDims, derivedStrides, ""},
          {4, "bias", DT::FLOAT, derivedDims, derivedStrides, ""}},
         {1, 2},
         {3, 4},
         {}},
        {"AcceptsHalf_IO",
         true,
         {{1, "x", DT::HALF, ioDims, ioStrides, ""},
          {2, "y", DT::HALF, ioDims, ioStrides, ""},
          {3, "scale", DT::FLOAT, derivedDims, derivedStrides, ""},
          {4, "bias", DT::FLOAT, derivedDims, derivedStrides, ""}},
         {1, 2},
         {3, 4},
         {}},
        {"AcceptsBfloat16_IO",
         true,
         {{1, "x", DT::BFLOAT16, ioDims, ioStrides, ""},
          {2, "y", DT::BFLOAT16, ioDims, ioStrides, ""},
          {3, "scale", DT::FLOAT, derivedDims, derivedStrides, ""},
          {4, "bias", DT::FLOAT, derivedDims, derivedStrides, ""}},
         {1, 2},
         {3, 4},
         {}},

        // Unhappy paths - invalid IO data type
        {"RejectsInvalidIoDataType",
         false,
         {{1, "x", DT::UINT8, ioDims, ioStrides, ""},
          {3, "scale", DT::FLOAT, derivedDims, derivedStrides, ""}},
         {1},
         {3},
         {}},

        // Unhappy paths - invalid affine data type (must be FLOAT)
        {"RejectsInvalidAffineDataType",
         false,
         {{1, "x", DT::FLOAT, ioDims, ioStrides, ""},
          {3, "scale", DT::HALF, derivedDims, derivedStrides, ""}},
         {1},
         {3},
         {}},
    };
}

// IO tensors match, affine/stat have derived shape
inline std::vector<TensorShapesComponentTestCase> getCheckTensorShapesSupportedTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;

    auto inferenceDims = shapes::INFERENCE_4D[2]; // {1, 3, 224, 224}
    auto inferenceStrides = hipdnn_data_sdk::utilities::generateStrides(
        inferenceDims, TensorLayout::NCHW.strideOrder);
    auto trainingDims = shapes::INFERENCE_4D[3]; // {2, 3, 224, 224}
    auto trainingStrides
        = hipdnn_data_sdk::utilities::generateStrides(trainingDims, TensorLayout::NCHW.strideOrder);
    auto mediumDims = shapes::INFERENCE_4D[1]; // {1, 3, 112, 112}
    auto mediumStrides
        = hipdnn_data_sdk::utilities::generateStrides(mediumDims, TensorLayout::NCHW.strideOrder);
    auto insufficientDims = shapes::INSUFFICIENT_SPATIAL_4D; // {1, 3, 1, 1}
    auto insufficientStrides = hipdnn_data_sdk::utilities::generateStrides(
        insufficientDims, TensorLayout::NCHW.strideOrder);

    auto derivedDimsInference
        = hipdnn_data_sdk::utilities::getDerivedShape(inferenceDims); // {1, 3, 1, 1}
    auto derivedStridesInference = hipdnn_data_sdk::utilities::generateStrides(
        derivedDimsInference, TensorLayout::NCHW.strideOrder);
    auto derivedDimsTraining
        = hipdnn_data_sdk::utilities::getDerivedShape(trainingDims); // {1, 3, 1, 1}
    auto derivedStridesTraining = hipdnn_data_sdk::utilities::generateStrides(
        derivedDimsTraining, TensorLayout::NCHW.strideOrder);

    auto wrongChannelDerivedDims = hipdnn_data_sdk::utilities::getDerivedShape(
        shapes::DIFFERENT_CHANNELS_4D); // {1, 5, 1, 1}
    auto wrongChannelDerivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        wrongChannelDerivedDims, TensorLayout::NCHW.strideOrder);

    return {
        // Happy paths - valid shapes for inference
        {"AcceptsValidInferenceShapes",
         true,
         {{1, "x", DT::FLOAT, inferenceDims, inferenceStrides, ""},
          {2, "y", DT::FLOAT, inferenceDims, inferenceStrides, ""},
          {3, "scale", DT::FLOAT, derivedDimsInference, derivedStridesInference, ""},
          {4, "bias", DT::FLOAT, derivedDimsInference, derivedStridesInference, ""}},
         {1, 2},
         {3, 4},
         {},
         false},

        // Happy paths - valid shapes for training with sufficient spatial
        {"AcceptsValidTrainingShapes",
         true,
         {{1, "x", DT::FLOAT, trainingDims, trainingStrides, ""},
          {2, "y", DT::FLOAT, trainingDims, trainingStrides, ""},
          {3, "scale", DT::FLOAT, derivedDimsTraining, derivedStridesTraining, ""}},
         {1, 2},
         {3},
         {},
         true},

        // Unhappy paths - inconsistent IO shapes
        {"RejectsInconsistentIoShapes",
         false,
         {{1, "x", DT::FLOAT, inferenceDims, inferenceStrides, ""},
          {2, "y", DT::FLOAT, mediumDims, mediumStrides, ""},
          {3, "scale", DT::FLOAT, derivedDimsInference, derivedStridesInference, ""}},
         {1, 2},
         {3},
         {},
         false},

        // Unhappy paths - wrong derived channel count
        {"RejectsWrongDerivedChannels",
         false,
         {{1, "x", DT::FLOAT, inferenceDims, inferenceStrides, ""},
          {3, "scale", DT::FLOAT, wrongChannelDerivedDims, wrongChannelDerivedStrides, ""}},
         {1},
         {3},
         {},
         false},

        // Unhappy paths - insufficient spatial for training (B × S ≤ 1)
        {"RejectsInsufficientSpatialForTraining",
         false,
         {{1, "x", DT::FLOAT, insufficientDims, insufficientStrides, ""},
          {2, "y", DT::FLOAT, insufficientDims, insufficientStrides, ""}},
         {1, 2},
         {},
         {},
         true},
    };
}

// --- Test Data Providers: Layer 3 (High-Level Validators) ---

inline std::vector<BatchnormInferenceConfigTestCase>
    getCheckBatchnormInferenceConfigSupportedTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;
    using UIDs = BnCommonTensorIds;

    std::vector<BatchnormInferenceConfigTestCase> cases;

    // Happy paths - all shapes × all 4D layouts × all valid type configurations
    for(const auto& dims : shapes::INFERENCE_4D)
    {
        for(const auto* layout : LAYOUTS_4D)
        {
            for(const auto& typeConfig : bn_type_configs::VALID)
            {
                cases.push_back(
                    {"AcceptsInference_" + generateName(dims, *layout) + "_" + toString(typeConfig),
                     true,
                     createBatchnormInferenceTensors(typeConfig, dims, *layout),
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::MEAN,
                     UIDs::INV_VARIANCE});
            }
        }
    }

    // Happy paths - 5D shapes × 5D layouts × all valid type configurations
    for(const auto& dims : shapes::INFERENCE_5D)
    {
        for(const auto* layout : LAYOUTS_5D)
        {
            for(const auto& typeConfig : bn_type_configs::VALID)
            {
                cases.push_back(
                    {"AcceptsInference_" + generateName(dims, *layout) + "_" + toString(typeConfig),
                     true,
                     createBatchnormInferenceTensors(typeConfig, dims, *layout),
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::MEAN,
                     UIDs::INV_VARIANCE});
            }
        }
    }

    // Unhappy paths - all invalid type configurations
    auto sampleDims = shapes::INFERENCE_4D[0];
    for(const auto& invalidConfig : type_configs::INVALID_ALL)
    {
        cases.push_back(
            {"RejectsInference_" + toString(invalidConfig),
             false,
             createBatchnormInferenceTensors(invalidConfig, sampleDims, TensorLayout::NCHW),
             UIDs::X,
             UIDs::Y,
             UIDs::SCALE,
             UIDs::BIAS,
             UIDs::MEAN,
             UIDs::INV_VARIANCE});
    }

    // Unhappy paths - mixed layouts (x is NCHW, y is NHWC)
    auto nchwStrides
        = hipdnn_data_sdk::utilities::generateStrides(sampleDims, TensorLayout::NCHW.strideOrder);
    auto nhwcStrides
        = hipdnn_data_sdk::utilities::generateStrides(sampleDims, TensorLayout::NHWC.strideOrder);
    auto sampleDerivedDims = hipdnn_data_sdk::utilities::getDerivedShape(sampleDims);
    auto sampleDerivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        sampleDerivedDims, TensorLayout::NCHW.strideOrder);

    std::vector<TensorConfig> mixedLayoutConfigs
        = {{UIDs::X, "x", DT::FLOAT, sampleDims, nchwStrides, ""},
           {UIDs::Y, "y", DT::FLOAT, sampleDims, nhwcStrides, ""},
           {UIDs::SCALE, "scale", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::BIAS, "bias", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::MEAN, "mean", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::INV_VARIANCE,
            "inv_variance",
            DT::FLOAT,
            sampleDerivedDims,
            sampleDerivedStrides,
            ""}};
    cases.push_back({"RejectsInference_MixedLayouts",
                     false,
                     mixedLayoutConfigs,
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::MEAN,
                     UIDs::INV_VARIANCE});

    return cases;
}

inline std::vector<BatchnormTrainingConfigTestCase>
    getCheckBatchnormTrainingConfigSupportedTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;
    using UIDs = BnTrainingTensorIds;

    std::vector<BatchnormTrainingConfigTestCase> cases;

    // ========================================================================
    // Happy Paths: Valid Type Configurations from bn_type_configs::VALID
    // ========================================================================

    // All training shapes × all layouts × all valid type configs × 2 variants
    for(const auto& typeConfig : bn_type_configs::VALID)
    {
        for(const auto& dims : shapes::TRAINING_4D)
        {
            for(const auto* layout : LAYOUTS_4D)
            {
                // With mean/variance
                cases.push_back(
                    {"AcceptsTraining_" + generateName(dims, *layout) + "_" + toString(typeConfig)
                         + "_WithMeanVar",
                     true,
                     createBatchnormTrainingTensors(typeConfig, dims, *layout, true, true),
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::EPSILON,
                     flatbuffers::Optional<int64_t>(UIDs::MEAN),
                     flatbuffers::Optional<int64_t>(UIDs::INV_VARIANCE)});
                // Without mean/variance
                cases.push_back(
                    {"AcceptsTraining_" + generateName(dims, *layout) + "_" + toString(typeConfig)
                         + "_WithoutMeanVar",
                     true,
                     createBatchnormTrainingTensors(typeConfig, dims, *layout, false, false),
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::EPSILON,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});
            }
        }
    }

    // Happy paths - 5D training shapes × all valid type configs
    for(const auto& typeConfig : bn_type_configs::VALID)
    {
        for(const auto& dims : shapes::TRAINING_5D)
        {
            for(const auto* layout : LAYOUTS_5D)
            {
                // With mean/variance
                cases.push_back(
                    {"AcceptsTraining_" + generateName(dims, *layout) + "_" + toString(typeConfig)
                         + "_WithMeanVar",
                     true,
                     createBatchnormTrainingTensors(typeConfig, dims, *layout, true, true),
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::EPSILON,
                     flatbuffers::Optional<int64_t>(UIDs::MEAN),
                     flatbuffers::Optional<int64_t>(UIDs::INV_VARIANCE)});
                // Without mean/variance
                cases.push_back(
                    {"AcceptsTraining_" + generateName(dims, *layout) + "_" + toString(typeConfig)
                         + "_WithoutMeanVar",
                     true,
                     createBatchnormTrainingTensors(typeConfig, dims, *layout, false, false),
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::EPSILON,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});
            }
        }
    }

    // ========================================================================
    // Unhappy Paths: Invalid Configurations
    // ========================================================================

    auto sampleTrainingDims = shapes::TRAINING_4D[0];

    // Unhappy paths - insufficient spatial dimensions (B × S ≤ 1)
    cases.push_back({"RejectsTraining_InsufficientSpatial_1x1",
                     false,
                     createBatchnormTrainingTensors(bn_type_configs::ALL_FLOAT,
                                                    shapes::INSUFFICIENT_SPATIAL_4D,
                                                    TensorLayout::NCHW,
                                                    false,
                                                    false),
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::EPSILON,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});

    // Unhappy paths - invalid IO data type (UINT8 instead of FLOAT)
    auto sampleTrainingStrides = hipdnn_data_sdk::utilities::generateStrides(
        sampleTrainingDims, TensorLayout::NCHW.strideOrder);
    auto derivedTrainingDims = hipdnn_data_sdk::utilities::getDerivedShape(sampleTrainingDims);
    auto derivedTrainingStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedTrainingDims, TensorLayout::NCHW.strideOrder);

    std::vector<TensorConfig> invalidTypeConfigs = {
        {UIDs::X, "x", DT::UINT8, sampleTrainingDims, sampleTrainingStrides, "", false, false, 0.0},
        {UIDs::Y, "y", DT::UINT8, sampleTrainingDims, sampleTrainingStrides, "", false, false, 0.0},
        {UIDs::SCALE,
         "scale",
         DT::FLOAT,
         derivedTrainingDims,
         derivedTrainingStrides,
         "",
         false,
         false,
         0.0},
        {UIDs::BIAS,
         "bias",
         DT::FLOAT,
         derivedTrainingDims,
         derivedTrainingStrides,
         "",
         false,
         false,
         0.0},
        {UIDs::EPSILON, "epsilon", DT::FLOAT, {1}, {1}, "", false, true, 1e-5}};
    cases.push_back({"RejectsTraining_InvalidIoDataType",
                     false,
                     invalidTypeConfigs,
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::EPSILON,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});

    // Unhappy paths - non-packed tensor (invalid strides)
    std::vector<TensorConfig> nonPackedConfigs = {
        {UIDs::X,
         "x",
         DT::FLOAT,
         sampleTrainingDims,
         {1000, 300, 20, 1},
         "",
         false,
         false,
         0.0}, // Non-packed! (intentionally invalid strides)
        {UIDs::Y, "y", DT::FLOAT, sampleTrainingDims, sampleTrainingStrides, "", false, false, 0.0},
        {UIDs::SCALE,
         "scale",
         DT::FLOAT,
         derivedTrainingDims,
         derivedTrainingStrides,
         "",
         false,
         false,
         0.0},
        {UIDs::BIAS,
         "bias",
         DT::FLOAT,
         derivedTrainingDims,
         derivedTrainingStrides,
         "",
         false,
         false,
         0.0},
        {UIDs::EPSILON, "epsilon", DT::FLOAT, {1}, {1}, "", false, true, 1e-5}};
    cases.push_back({"RejectsTraining_NonPackedTensor",
                     false,
                     nonPackedConfigs,
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::EPSILON,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});

    // Unhappy paths - all invalid type configurations
    for(const auto& invalidConfig : type_configs::INVALID_ALL)
    {
        const bool hasInvalidStatType = (invalidConfig.stat != DT::FLOAT);

        // Always test with mean/variance outputs (validates all types)
        cases.push_back({"RejectsTraining_" + toString(invalidConfig) + "_WithMeanVar",
                         false,
                         createBatchnormTrainingTensors(
                             invalidConfig, sampleTrainingDims, TensorLayout::NCHW, true, true),
                         UIDs::X,
                         UIDs::Y,
                         UIDs::SCALE,
                         UIDs::BIAS,
                         UIDs::EPSILON,
                         flatbuffers::Optional<int64_t>(UIDs::MEAN),
                         flatbuffers::Optional<int64_t>(UIDs::INV_VARIANCE)});

        // Only test without mean/variance if stat type is valid (FLOAT)
        // (Can't validate stat type if no stat tensors exist)
        if(!hasInvalidStatType)
        {
            cases.push_back(
                {"RejectsTraining_" + toString(invalidConfig) + "_NoMeanVar",
                 false,
                 createBatchnormTrainingTensors(
                     invalidConfig, sampleTrainingDims, TensorLayout::NCHW, false, false),
                 UIDs::X,
                 UIDs::Y,
                 UIDs::SCALE,
                 UIDs::BIAS,
                 UIDs::EPSILON,
                 flatbuffers::nullopt,
                 flatbuffers::nullopt});
        }
    }

    return cases;
}

inline std::vector<BatchnormBackwardConfigTestCase>
    getCheckBatchnormBackwardConfigSupportedTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;
    using UIDs = BnBackwardTensorIds;

    std::vector<BatchnormBackwardConfigTestCase> cases;

    // Happy paths - all inference shapes × all 4D layouts × all valid type configs × 2 variants
    for(const auto& dims : shapes::INFERENCE_4D)
    {
        for(const auto* layout : LAYOUTS_4D)
        {
            for(const auto& typeConfig : bn_type_configs::VALID)
            {
                // With optionals
                cases.push_back(
                    {"AcceptsBackward_" + generateName(dims, *layout) + "_" + toString(typeConfig)
                         + "_WithOptionals",
                     true,
                     createBatchnormBackwardTensors(typeConfig, dims, *layout, true, true),
                     UIDs::X,
                     UIDs::DY,
                     UIDs::DX,
                     UIDs::SCALE,
                     UIDs::DSCALE,
                     UIDs::DBIAS,
                     flatbuffers::Optional<int64_t>(UIDs::MEAN),
                     flatbuffers::Optional<int64_t>(UIDs::INV_VARIANCE)});
                // Without optionals
                cases.push_back(
                    {"AcceptsBackward_" + generateName(dims, *layout) + "_" + toString(typeConfig)
                         + "_WithoutOptionals",
                     true,
                     createBatchnormBackwardTensors(typeConfig, dims, *layout, false, false),
                     UIDs::X,
                     UIDs::DY,
                     UIDs::DX,
                     UIDs::SCALE,
                     UIDs::DSCALE,
                     UIDs::DBIAS,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});
            }
        }
    }

    // Happy paths - 5D inference shapes × 5D layouts × all valid type configs
    for(const auto& dims : shapes::INFERENCE_5D)
    {
        for(const auto* layout : LAYOUTS_5D)
        {
            for(const auto& typeConfig : bn_type_configs::VALID)
            {
                // With optionals
                cases.push_back(
                    {"AcceptsBackward_" + generateName(dims, *layout) + "_" + toString(typeConfig)
                         + "_WithOptionals",
                     true,
                     createBatchnormBackwardTensors(typeConfig, dims, *layout, true, true),
                     UIDs::X,
                     UIDs::DY,
                     UIDs::DX,
                     UIDs::SCALE,
                     UIDs::DSCALE,
                     UIDs::DBIAS,
                     flatbuffers::Optional<int64_t>(UIDs::MEAN),
                     flatbuffers::Optional<int64_t>(UIDs::INV_VARIANCE)});
                // Without optionals
                cases.push_back(
                    {"AcceptsBackward_" + generateName(dims, *layout) + "_" + toString(typeConfig)
                         + "_WithoutOptionals",
                     true,
                     createBatchnormBackwardTensors(typeConfig, dims, *layout, false, false),
                     UIDs::X,
                     UIDs::DY,
                     UIDs::DX,
                     UIDs::SCALE,
                     UIDs::DSCALE,
                     UIDs::DBIAS,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});
            }
        }
    }

    // Unhappy paths - all invalid type configurations
    auto sampleDims = shapes::INFERENCE_4D[0];
    for(const auto& invalidConfig : type_configs::INVALID_ALL)
    {
        const bool hasInvalidStatType = (invalidConfig.stat != DT::FLOAT);

        // Always test with mean/variance inputs (validates all types)
        cases.push_back({"RejectsBackward_" + toString(invalidConfig) + "_WithMeanVar",
                         false,
                         createBatchnormBackwardTensors(
                             invalidConfig, sampleDims, TensorLayout::NCHW, true, true),
                         UIDs::X,
                         UIDs::DY,
                         UIDs::DX,
                         UIDs::SCALE,
                         UIDs::DSCALE,
                         UIDs::DBIAS,
                         flatbuffers::Optional<int64_t>(UIDs::MEAN),
                         flatbuffers::Optional<int64_t>(UIDs::INV_VARIANCE)});

        // Only test without mean/variance if stat type is valid (FLOAT)
        // (Can't validate stat type if no stat tensors exist)
        if(!hasInvalidStatType)
        {
            cases.push_back({"RejectsBackward_" + toString(invalidConfig) + "_NoMeanVar",
                             false,
                             createBatchnormBackwardTensors(
                                 invalidConfig, sampleDims, TensorLayout::NCHW, false, false),
                             UIDs::X,
                             UIDs::DY,
                             UIDs::DX,
                             UIDs::SCALE,
                             UIDs::DSCALE,
                             UIDs::DBIAS,
                             flatbuffers::nullopt,
                             flatbuffers::nullopt});
        }
    }

    // Unhappy paths - inconsistent IO shapes
    auto sampleStrides
        = hipdnn_data_sdk::utilities::generateStrides(sampleDims, TensorLayout::NCHW.strideOrder);
    auto sampleDerivedDims = hipdnn_data_sdk::utilities::getDerivedShape(sampleDims);
    auto sampleDerivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        sampleDerivedDims, TensorLayout::NCHW.strideOrder);
    auto mediumDims = shapes::INFERENCE_4D[1]; // {1, 3, 112, 112}
    auto mediumStrides
        = hipdnn_data_sdk::utilities::generateStrides(mediumDims, TensorLayout::NCHW.strideOrder);

    std::vector<TensorConfig> inconsistentShapes
        = {{UIDs::X, "x", DT::FLOAT, sampleDims, sampleStrides, ""},
           {UIDs::DY, "dy", DT::FLOAT, mediumDims, mediumStrides, ""}, // Different!
           {UIDs::DX, "dx", DT::FLOAT, sampleDims, sampleStrides, ""},
           {UIDs::SCALE, "scale", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::DSCALE, "dscale", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::DBIAS, "dbias", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""}};
    cases.push_back({"RejectsBackward_InconsistentShapes",
                     false,
                     inconsistentShapes,
                     UIDs::X,
                     UIDs::DY,
                     UIDs::DX,
                     UIDs::SCALE,
                     UIDs::DSCALE,
                     UIDs::DBIAS,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});

    // Unhappy paths - non-packed tensor
    std::vector<TensorConfig> nonPackedConfigs
        = {{UIDs::X, "x", DT::FLOAT, sampleDims, sampleStrides, ""},
           {UIDs::DY,
            "dy",
            DT::FLOAT,
            sampleDims,
            {200000, 60000, 250, 1},
            ""}, // Non-packed! (intentionally invalid strides)
           {UIDs::DX, "dx", DT::FLOAT, sampleDims, sampleStrides, ""},
           {UIDs::SCALE, "scale", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::DSCALE, "dscale", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::DBIAS, "dbias", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""}};
    cases.push_back({"RejectsBackward_NonPackedTensor",
                     false,
                     nonPackedConfigs,
                     UIDs::X,
                     UIDs::DY,
                     UIDs::DX,
                     UIDs::SCALE,
                     UIDs::DSCALE,
                     UIDs::DBIAS,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});

    return cases;
}

inline std::vector<BatchnormFusedBackwardConfigTestCase>
    getCheckBatchnormFusedBackwardConfigSupportedTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;
    using UIDs = BnFusedTensorIds;

    std::vector<BatchnormFusedBackwardConfigTestCase> cases;

    // Supported backward activation modes (from atomic validation tests)
    using PM = hipdnn_data_sdk::data_objects::PointwiseMode;
    const std::vector<PM> supportedBwdActivations = {PM::IDENTITY, PM::RELU_BWD};

    // Happy paths - all inference shapes × all 4D layouts × all valid type configs × all supported activations
    for(const auto& dims : shapes::INFERENCE_4D)
    {
        for(const auto* layout : LAYOUTS_4D)
        {
            for(const auto& typeConfig : bn_type_configs::VALID)
            {
                for(const auto& activMode : supportedBwdActivations)
                {
                    cases.push_back({"AcceptsFused_" + generateName(dims, *layout) + "_"
                                         + toString(typeConfig) + "_"
                                         + activationModeToString(activMode),
                                     true,
                                     createBatchnormFusedBackwardTensors(typeConfig, dims, *layout),
                                     UIDs::X,
                                     UIDs::SCALE,
                                     UIDs::BIAS,
                                     UIDs::MEAN,
                                     UIDs::INV_VARIANCE,
                                     UIDs::DY,
                                     UIDs::DX,
                                     UIDs::DSCALE,
                                     UIDs::DBIAS,
                                     UIDs::BN_Y_VIRTUAL,
                                     UIDs::DX_DRELU_VIRTUAL,
                                     activMode});
                }
            }
        }
    }

    // Happy paths - 5D inference shapes × 5D layouts × all valid type configs × all supported activations
    for(const auto& dims : shapes::INFERENCE_5D)
    {
        for(const auto* layout : LAYOUTS_5D)
        {
            for(const auto& typeConfig : bn_type_configs::VALID)
            {
                for(const auto& activMode : supportedBwdActivations)
                {
                    cases.push_back({"AcceptsFused_" + generateName(dims, *layout) + "_"
                                         + toString(typeConfig) + "_"
                                         + activationModeToString(activMode),
                                     true,
                                     createBatchnormFusedBackwardTensors(typeConfig, dims, *layout),
                                     UIDs::X,
                                     UIDs::SCALE,
                                     UIDs::BIAS,
                                     UIDs::MEAN,
                                     UIDs::INV_VARIANCE,
                                     UIDs::DY,
                                     UIDs::DX,
                                     UIDs::DSCALE,
                                     UIDs::DBIAS,
                                     UIDs::BN_Y_VIRTUAL,
                                     UIDs::DX_DRELU_VIRTUAL,
                                     activMode});
                }
            }
        }
    }

    // Unhappy paths - all invalid type configurations
    auto sampleDims = shapes::INFERENCE_4D[0];
    for(const auto& invalidConfig : type_configs::INVALID_ALL)
    {
        cases.push_back(
            {"RejectsFused_" + toString(invalidConfig),
             false,
             createBatchnormFusedBackwardTensors(invalidConfig, sampleDims, TensorLayout::NCHW),
             UIDs::X,
             UIDs::SCALE,
             UIDs::BIAS,
             UIDs::MEAN,
             UIDs::INV_VARIANCE,
             UIDs::DY,
             UIDs::DX,
             UIDs::DSCALE,
             UIDs::DBIAS,
             UIDs::BN_Y_VIRTUAL,
             UIDs::DX_DRELU_VIRTUAL,
             PM::RELU_BWD});
    }

    // Unhappy paths - mixed layouts
    auto sampleStrides
        = hipdnn_data_sdk::utilities::generateStrides(sampleDims, TensorLayout::NCHW.strideOrder);
    auto sampleDerivedDims = hipdnn_data_sdk::utilities::getDerivedShape(sampleDims);
    auto sampleDerivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        sampleDerivedDims, TensorLayout::NCHW.strideOrder);
    auto nhwcStrides
        = hipdnn_data_sdk::utilities::generateStrides(sampleDims, TensorLayout::NHWC.strideOrder);

    std::vector<TensorConfig> mixedLayoutConfigs
        = {{UIDs::X, "x", DT::FLOAT, sampleDims, sampleStrides, ""},
           {UIDs::SCALE, "scale", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::BIAS, "bias", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::MEAN, "mean", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::INV_VARIANCE,
            "inv_variance",
            DT::FLOAT,
            sampleDerivedDims,
            sampleDerivedStrides,
            ""},
           {UIDs::DY, "dy", DT::FLOAT, sampleDims, nhwcStrides, ""}, // NHWC - different!
           {UIDs::DX, "dx", DT::FLOAT, sampleDims, sampleStrides, ""},
           {UIDs::DSCALE, "dscale", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::DBIAS, "dbias", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::BN_Y_VIRTUAL, "BN_Y", DT::FLOAT, sampleDims, sampleStrides, "", true},
           {UIDs::DX_DRELU_VIRTUAL, "DX_drelu", DT::FLOAT, sampleDims, sampleStrides, "", true}};
    cases.push_back({"RejectsFused_MixedLayouts",
                     false,
                     mixedLayoutConfigs,
                     BnFusedTensorIds::X,
                     BnFusedTensorIds::SCALE,
                     BnFusedTensorIds::BIAS,
                     BnFusedTensorIds::MEAN,
                     BnFusedTensorIds::INV_VARIANCE,
                     BnFusedTensorIds::DY,
                     BnFusedTensorIds::DX,
                     BnFusedTensorIds::DSCALE,
                     BnFusedTensorIds::DBIAS,
                     BnFusedTensorIds::BN_Y_VIRTUAL,
                     BnFusedTensorIds::DX_DRELU_VIRTUAL,
                     PM::RELU_BWD});

    // Unhappy paths - unsupported activation: SIGMOID_BWD
    cases.push_back({"RejectsFused_UnsupportedActivation_SigmoidBwd",
                     false,
                     createBatchnormFusedBackwardTensors(
                         bn_type_configs::ALL_FLOAT, sampleDims, TensorLayout::NCHW),
                     BnFusedTensorIds::X,
                     BnFusedTensorIds::SCALE,
                     BnFusedTensorIds::BIAS,
                     BnFusedTensorIds::MEAN,
                     BnFusedTensorIds::INV_VARIANCE,
                     BnFusedTensorIds::DY,
                     BnFusedTensorIds::DX,
                     BnFusedTensorIds::DSCALE,
                     BnFusedTensorIds::DBIAS,
                     BnFusedTensorIds::BN_Y_VIRTUAL,
                     BnFusedTensorIds::DX_DRELU_VIRTUAL,
                     PM::SIGMOID_BWD});

    // Unhappy paths - unsupported activation: TANH_BWD
    cases.push_back({"RejectsFused_UnsupportedActivation_TanhBwd",
                     false,
                     createBatchnormFusedBackwardTensors(
                         bn_type_configs::ALL_FLOAT, sampleDims, TensorLayout::NCHW),
                     BnFusedTensorIds::X,
                     BnFusedTensorIds::SCALE,
                     BnFusedTensorIds::BIAS,
                     BnFusedTensorIds::MEAN,
                     BnFusedTensorIds::INV_VARIANCE,
                     BnFusedTensorIds::DY,
                     BnFusedTensorIds::DX,
                     BnFusedTensorIds::DSCALE,
                     BnFusedTensorIds::DBIAS,
                     BnFusedTensorIds::BN_Y_VIRTUAL,
                     BnFusedTensorIds::DX_DRELU_VIRTUAL,
                     PM::TANH_BWD});

    // Unhappy paths - unsupported activation: RELU_FWD (wrong direction)
    cases.push_back({"RejectsFused_UnsupportedActivation_ReluFwdInBwdContext",
                     false,
                     createBatchnormFusedBackwardTensors(
                         bn_type_configs::ALL_FLOAT, sampleDims, TensorLayout::NCHW),
                     BnFusedTensorIds::X,
                     BnFusedTensorIds::SCALE,
                     BnFusedTensorIds::BIAS,
                     BnFusedTensorIds::MEAN,
                     BnFusedTensorIds::INV_VARIANCE,
                     BnFusedTensorIds::DY,
                     BnFusedTensorIds::DX,
                     BnFusedTensorIds::DSCALE,
                     BnFusedTensorIds::DBIAS,
                     BnFusedTensorIds::BN_Y_VIRTUAL,
                     BnFusedTensorIds::DX_DRELU_VIRTUAL,
                     PM::RELU_FWD});

    return cases;
}

inline std::vector<ActivationModeTestCase> getCheckBatchnormFwdActivationModeSupportedTestCases()
{
    return {
        // Happy paths - supported activation modes
        {"AcceptsIdentity",
         true,
         hipdnn_data_sdk::data_objects::PointwiseMode::IDENTITY,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
        {"AcceptsRelu",
         true,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
        {"AcceptsClippedRelu",
         true,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD,
         flatbuffers::nullopt,
         flatbuffers::Optional<double>(6.0),
         flatbuffers::nullopt},
        {"AcceptsClamp",
         true,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD,
         flatbuffers::Optional<double>(0.0),
         flatbuffers::Optional<double>(6.0),
         flatbuffers::nullopt},

        // Unhappy paths - unsupported activation modes
        {"RejectsLeakyRelu",
         false,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::Optional<double>(0.01)},
        {"RejectsSigmoid",
         false,
         hipdnn_data_sdk::data_objects::PointwiseMode::SIGMOID_FWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
        {"RejectsTanh",
         false,
         hipdnn_data_sdk::data_objects::PointwiseMode::TANH_FWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
        {"RejectsReluBwdInFwdContext",
         false,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
    };
}

inline std::vector<ActivationModeTestCase> getCheckBatchnormBwdActivationModeSupportedTestCases()
{
    return {
        // Happy paths - supported activation modes
        {"AcceptsIdentityBwd",
         true,
         hipdnn_data_sdk::data_objects::PointwiseMode::IDENTITY,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
        {"AcceptsReluBwd",
         true,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
        {"AcceptsClippedReluBwd",
         true,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD,
         flatbuffers::nullopt,
         flatbuffers::Optional<double>(6.0),
         flatbuffers::nullopt},
        {"AcceptsClampBwd",
         true,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD,
         flatbuffers::Optional<double>(0.0),
         flatbuffers::Optional<double>(6.0),
         flatbuffers::nullopt},

        // Unhappy paths - unsupported activation modes
        {"RejectsLeakyReluBwd",
         false,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::Optional<double>(0.01)},
        {"RejectsSigmoidBwd",
         false,
         hipdnn_data_sdk::data_objects::PointwiseMode::SIGMOID_BWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
        {"RejectsTanhBwd",
         false,
         hipdnn_data_sdk::data_objects::PointwiseMode::TANH_BWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
        {"RejectsReluFwdInBwdContext",
         false,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
    };
}

} // namespace

// --- Test Classes: Layer 1 (Atomic Validators) ---

class TestValidateDimensionCount : public ::testing::TestWithParam<DimensionCountTestCase>
{
};

TEST_P(TestValidateDimensionCount, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ validators::validateDimensionCount(tc.numDims); });
    }
    else
    {
        EXPECT_THROW(
            { validators::validateDimensionCount(tc.numDims); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateDimensionCount,
                         testing::ValuesIn(getValidateDimensionCountTestCases()));

class TestValidateConsistentDimensions
    : public ::testing::TestWithParam<TensorDescriptorListTestCase>
{
};

TEST_P(TestValidateConsistentDimensions, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    std::vector<BatchnormTensorDescriptor> tensors;
    tensors.reserve(tc.tensorDims.size());
    for(size_t i = 0; i < tc.tensorDims.size(); ++i)
    {
        flatbuffers::FlatBufferBuilder builder;
        auto tensorOffset = hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            static_cast<int64_t>(i + 1),
            ("tensor_" + std::to_string(i + 1)).c_str(),
            hipdnn_data_sdk::data_objects::DataType::FLOAT,
            &tc.tensorStrides[i],
            &tc.tensorDims[i]);
        builder.Finish(tensorOffset);

        const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::TensorAttributes>(
            builder.GetBufferPointer());
        tensors.emplace_back(attr);
    }

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ validators::validateConsistentDimensions(tensors); });
    }
    else
    {
        EXPECT_THROW(
            { validators::validateConsistentDimensions(tensors); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateConsistentDimensions,
                         testing::ValuesIn(getValidateConsistentDimensionsTestCases()));

class TestValidatePackedTensors : public ::testing::TestWithParam<TensorDescriptorListTestCase>
{
};

TEST_P(TestValidatePackedTensors, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    std::vector<BatchnormTensorDescriptor> tensors;
    tensors.reserve(tc.tensorDims.size());
    for(size_t i = 0; i < tc.tensorDims.size(); ++i)
    {
        flatbuffers::FlatBufferBuilder builder;
        auto tensorOffset = hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            static_cast<int64_t>(i + 1),
            ("tensor_" + std::to_string(i + 1)).c_str(),
            hipdnn_data_sdk::data_objects::DataType::FLOAT,
            &tc.tensorStrides[i],
            &tc.tensorDims[i]);
        builder.Finish(tensorOffset);

        const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::TensorAttributes>(
            builder.GetBufferPointer());
        tensors.emplace_back(attr);
    }

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ validators::validatePackedTensors(tensors); });
    }
    else
    {
        EXPECT_THROW(
            { validators::validatePackedTensors(tensors); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidatePackedTensors,
                         testing::ValuesIn(getValidatePackedTensorsTestCases()));

class TestValidateSupportedLayout : public ::testing::TestWithParam<SupportedLayoutTestCase>
{
};

TEST_P(TestValidateSupportedLayout, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ validators::validateSupportedLayout(tc.strideOrder, tc.numDims); });
    }
    else
    {
        EXPECT_THROW(
            { validators::validateSupportedLayout(tc.strideOrder, tc.numDims); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateSupportedLayout,
                         testing::ValuesIn(getValidateSupportedLayoutTestCases()));
class TestValidateConsistentLayouts : public ::testing::TestWithParam<TensorDescriptorListTestCase>
{
};

TEST_P(TestValidateConsistentLayouts, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    std::vector<BatchnormTensorDescriptor> tensors;
    tensors.reserve(tc.tensorDims.size());
    for(size_t i = 0; i < tc.tensorDims.size(); ++i)
    {
        flatbuffers::FlatBufferBuilder builder;
        auto tensorOffset = hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            static_cast<int64_t>(i + 1),
            ("tensor_" + std::to_string(i + 1)).c_str(),
            hipdnn_data_sdk::data_objects::DataType::FLOAT,
            &tc.tensorStrides[i],
            &tc.tensorDims[i]);
        builder.Finish(tensorOffset);

        const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::TensorAttributes>(
            builder.GetBufferPointer());
        tensors.emplace_back(attr);
    }

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ validators::validateConsistentLayouts(tensors); });
    }
    else
    {
        EXPECT_THROW(
            { validators::validateConsistentLayouts(tensors); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateConsistentLayouts,
                         testing::ValuesIn(getValidateConsistentLayoutsTestCases()));

class TestValidateDataTypeIsSupported : public ::testing::TestWithParam<DataTypeIsSupportedTestCase>
{
};

TEST_P(TestValidateDataTypeIsSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({
            validators::validateDataTypeIsSupported(
                tc.dataType, tc.allowedTypes, "Test error message");
        });
    }
    else
    {
        EXPECT_THROW(
            {
                validators::validateDataTypeIsSupported(
                    tc.dataType, tc.allowedTypes, "Test error message");
            },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateDataTypeIsSupported,
                         testing::ValuesIn(getValidateDataTypeIsSupportedTestCases()));

class TestValidateConsistentDataTypes : public ::testing::TestWithParam<ConsistentDataTypesTestCase>
{
};

TEST_P(TestValidateConsistentDataTypes, ValidatesCorrectly)
{
    const auto& tc = GetParam();
    auto [builder, tensorMap] = buildTensorMapFromConfigs(tc.tensorConfigs);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({
            validators::validateConsistentDataTypes(
                tc.tensorIds, tensorMap, tc.allowedTypes, "Type error", "Consistency error");
        });
    }
    else
    {
        EXPECT_THROW(
            {
                validators::validateConsistentDataTypes(
                    tc.tensorIds, tensorMap, tc.allowedTypes, "Type error", "Consistency error");
            },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateConsistentDataTypes,
                         testing::ValuesIn(getValidateConsistentDataTypesTestCases()));

class TestValidateFixedDataType : public ::testing::TestWithParam<FixedDataTypeTestCase>
{
};

TEST_P(TestValidateFixedDataType, ValidatesCorrectly)
{
    const auto& tc = GetParam();
    auto [builder, tensorMap] = buildTensorMapFromConfigs(tc.tensorConfigs);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({
            validators::validateFixedDataType(
                tc.tensorIds, tensorMap, tc.expectedType, "Type mismatch error");
        });
    }
    else
    {
        EXPECT_THROW(
            {
                validators::validateFixedDataType(
                    tc.tensorIds, tensorMap, tc.expectedType, "Type mismatch error");
            },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateFixedDataType,
                         testing::ValuesIn(getValidateFixedDataTypeTestCases()));

class TestValidateConsistentShapes : public ::testing::TestWithParam<ConsistentShapesTestCase>
{
};

TEST_P(TestValidateConsistentShapes, ValidatesCorrectly)
{
    const auto& tc = GetParam();
    auto [builder, tensorMap] = buildTensorMapFromConfigs(tc.tensorConfigs);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({
            validators::validateConsistentShapes(
                tc.tensorIds, tensorMap, tc.referenceShape, "Shape mismatch error");
        });
    }
    else
    {
        EXPECT_THROW(
            {
                validators::validateConsistentShapes(
                    tc.tensorIds, tensorMap, tc.referenceShape, "Shape mismatch error");
            },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateConsistentShapes,
                         testing::ValuesIn(getValidateConsistentShapesTestCases()));

class TestValidateSpatialDimensions : public ::testing::TestWithParam<SpatialDimensionsTestCase>
{
};

TEST_P(TestValidateSpatialDimensions, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ validators::validateSpatialDimensions(tc.ioDims); });
    }
    else
    {
        EXPECT_THROW(
            { validators::validateSpatialDimensions(tc.ioDims); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateSpatialDimensions,
                         testing::ValuesIn(getValidateSpatialDimensionsTestCases()));

// Note: validatePeerStatsNotPopulated is tested through integration tests below
// (RejectsBatchnormFwdTrainingWithPeerStats, RejectsBatchnormBackwardWithPeerStats)

// --- Test Classes: Layer 2 (Component Validators) ---

class TestCheckTensorLayoutsAndDimsSupported
    : public ::testing::TestWithParam<TensorLayoutsAndDimsTestCase>
{
};

TEST_P(TestCheckTensorLayoutsAndDimsSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();
    auto [builder, tensorMap] = buildTensorMapFromConfigs(tc.tensorConfigs);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ checkTensorLayoutsAndDimsSupported(tensorMap); });
    }
    else
    {
        EXPECT_THROW(
            { checkTensorLayoutsAndDimsSupported(tensorMap); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestCheckTensorLayoutsAndDimsSupported,
                         testing::ValuesIn(getCheckTensorLayoutsAndDimsSupportedTestCases()));

class TestCheckTensorDataTypesSupported
    : public ::testing::TestWithParam<TensorDataTypesComponentTestCase>
{
};

TEST_P(TestCheckTensorDataTypesSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();
    auto [builder, tensorMap] = buildTensorMapFromConfigs(tc.tensorConfigs);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({
            checkTensorDataTypesSupported(
                tc.ioTensorIds, tc.affineTensorIds, tc.statTensorIds, tensorMap);
        });
    }
    else
    {
        EXPECT_THROW(
            {
                checkTensorDataTypesSupported(
                    tc.ioTensorIds, tc.affineTensorIds, tc.statTensorIds, tensorMap);
            },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestCheckTensorDataTypesSupported,
                         testing::ValuesIn(getCheckTensorDataTypesSupportedTestCases()));

class TestCheckTensorShapesSupported
    : public ::testing::TestWithParam<TensorShapesComponentTestCase>
{
};

TEST_P(TestCheckTensorShapesSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();
    auto [builder, tensorMap] = buildTensorMapFromConfigs(tc.tensorConfigs);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({
            checkTensorShapesSupported(
                tc.ioTensorIds, tc.affineTensorIds, tc.statTensorIds, tensorMap, tc.isTraining);
        });
    }
    else
    {
        EXPECT_THROW(
            {
                checkTensorShapesSupported(
                    tc.ioTensorIds, tc.affineTensorIds, tc.statTensorIds, tensorMap, tc.isTraining);
            },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestCheckTensorShapesSupported,
                         testing::ValuesIn(getCheckTensorShapesSupportedTestCases()));

// --- Test Classes: Layer 3 (High-Level Configuration Validators) ---

class TestCheckBatchnormInferenceConfigSupported
    : public ::testing::TestWithParam<BatchnormInferenceConfigTestCase>
{
};

TEST_P(TestCheckBatchnormInferenceConfigSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    auto builder = buildBatchnormInferenceGraph(
        tc.tensorConfigs, tc.xUid, tc.yUid, tc.scaleUid, tc.biasUid, tc.meanUid, tc.invVarianceUid);
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormInferenceAttributes();
    ASSERT_NE(attrs, nullptr);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ checkBatchnormTensorConfigSupported(*attrs, graph.getTensorMap()); });
    }
    else
    {
        EXPECT_THROW(
            { checkBatchnormTensorConfigSupported(*attrs, graph.getTensorMap()); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestCheckBatchnormInferenceConfigSupported,
                         testing::ValuesIn(getCheckBatchnormInferenceConfigSupportedTestCases()));

class TestCheckBatchnormTrainingConfigSupported
    : public ::testing::TestWithParam<BatchnormTrainingConfigTestCase>
{
};

TEST_P(TestCheckBatchnormTrainingConfigSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    auto builder = buildBatchnormTrainingGraph(tc.tensorConfigs,
                                               tc.xUid,
                                               tc.yUid,
                                               tc.scaleUid,
                                               tc.biasUid,
                                               tc.epsilonUid,
                                               tc.meanUid,
                                               tc.invVarianceUid);
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ checkBatchnormTensorConfigSupported(*attrs, graph.getTensorMap()); });
    }
    else
    {
        EXPECT_THROW(
            { checkBatchnormTensorConfigSupported(*attrs, graph.getTensorMap()); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestCheckBatchnormTrainingConfigSupported,
                         testing::ValuesIn(getCheckBatchnormTrainingConfigSupportedTestCases()));

class TestCheckBatchnormBackwardConfigSupported
    : public ::testing::TestWithParam<BatchnormBackwardConfigTestCase>
{
};

TEST_P(TestCheckBatchnormBackwardConfigSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    auto builder = buildBatchnormBackwardGraph(tc.tensorConfigs,
                                               tc.xUid,
                                               tc.dyUid,
                                               tc.dxUid,
                                               tc.scaleUid,
                                               tc.dscaleUid,
                                               tc.dbiasUid,
                                               tc.meanUid,
                                               tc.invVarianceUid);
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ checkBatchnormTensorConfigSupported(*attrs, graph.getTensorMap()); });
    }
    else
    {
        EXPECT_THROW(
            { checkBatchnormTensorConfigSupported(*attrs, graph.getTensorMap()); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestCheckBatchnormBackwardConfigSupported,
                         testing::ValuesIn(getCheckBatchnormBackwardConfigSupportedTestCases()));

class TestCheckBatchnormFusedBackwardConfigSupported
    : public ::testing::TestWithParam<BatchnormFusedBackwardConfigTestCase>
{
};

TEST_P(TestCheckBatchnormFusedBackwardConfigSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    auto builder = buildBatchnormFusedBackwardGraph(tc.tensorConfigs,
                                                    tc.xUid,
                                                    tc.scaleUid,
                                                    tc.biasUid,
                                                    tc.meanUid,
                                                    tc.invVarianceUid,
                                                    tc.dyUid,
                                                    tc.dxUid,
                                                    tc.dscaleUid,
                                                    tc.dbiasUid,
                                                    tc.bnYVirtualUid,
                                                    tc.dxDreluVirtualUid,
                                                    tc.activationMode);
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnInfNode = graph.getNode(0);
    const auto& actNode = graph.getNode(1);
    const auto& bnBwdNode = graph.getNode(2);

    auto* bnInfAttrs = bnInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* actAttrs = actNode.attributes_as_PointwiseAttributes();
    auto* bnBwdAttrs = bnBwdNode.attributes_as_BatchnormBackwardAttributes();

    ASSERT_NE(bnInfAttrs, nullptr);
    ASSERT_NE(actAttrs, nullptr);
    ASSERT_NE(bnBwdAttrs, nullptr);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({
            checkBatchnormTensorConfigSupported(
                *bnInfAttrs, *actAttrs, *bnBwdAttrs, graph.getTensorMap());
        });
    }
    else
    {
        EXPECT_THROW(
            {
                checkBatchnormTensorConfigSupported(
                    *bnInfAttrs, *actAttrs, *bnBwdAttrs, graph.getTensorMap());
            },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllCases,
    TestCheckBatchnormFusedBackwardConfigSupported,
    testing::ValuesIn(getCheckBatchnormFusedBackwardConfigSupportedTestCases()));

class TestCheckBatchnormFwdActivationModeSupported
    : public ::testing::TestWithParam<ActivationModeTestCase>
{
};

TEST_P(TestCheckBatchnormFwdActivationModeSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    flatbuffers::FlatBufferBuilder builder;
    auto actAttr = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(builder,
                                                                            tc.mode,
                                                                            tc.reluLowerClip,
                                                                            tc.reluUpperClip,
                                                                            tc.reluLowerClipSlope,
                                                                            flatbuffers::nullopt,
                                                                            1,
                                                                            flatbuffers::nullopt,
                                                                            flatbuffers::nullopt,
                                                                            2);
    builder.Finish(actAttr);

    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ checkBatchnormFwdActivationModeSupported(*attr); });
    }
    else
    {
        EXPECT_THROW(
            { checkBatchnormFwdActivationModeSupported(*attr); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestCheckBatchnormFwdActivationModeSupported,
                         testing::ValuesIn(getCheckBatchnormFwdActivationModeSupportedTestCases()));

class TestCheckBatchnormBwdActivationModeSupported
    : public ::testing::TestWithParam<ActivationModeTestCase>
{
};

TEST_P(TestCheckBatchnormBwdActivationModeSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    flatbuffers::FlatBufferBuilder builder;
    auto actAttr = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(builder,
                                                                            tc.mode,
                                                                            tc.reluLowerClip,
                                                                            tc.reluUpperClip,
                                                                            tc.reluLowerClipSlope,
                                                                            flatbuffers::nullopt,
                                                                            1,
                                                                            flatbuffers::nullopt,
                                                                            flatbuffers::nullopt,
                                                                            2);
    builder.Finish(actAttr);

    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ checkBatchnormBwdActivationModeSupported(*attr); });
    }
    else
    {
        EXPECT_THROW(
            { checkBatchnormBwdActivationModeSupported(*attr); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestCheckBatchnormBwdActivationModeSupported,
                         testing::ValuesIn(getCheckBatchnormBwdActivationModeSupportedTestCases()));

// --- Integration Tests: Peer Stats Validation ---

TEST(TestBatchnormApplicabilityChecks, RejectsBatchnormFwdTrainingWithPeerStats)
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;
    using UIDs = BnTrainingTensorIds;

    // Create valid tensor configs
    const auto& dims = shapes::TRAINING_4D[0];
    auto configs = createBatchnormTrainingTensors(
        bn_type_configs::ALL_FLOAT, dims, TensorLayout::NCHW, false, false);

    flatbuffers::FlatBufferBuilder builder;

    // Build tensor attributes
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>> tensorOffsets;
    for(const auto& cfg : configs)
    {
        if(cfg.isPassByValue)
        {
            hipdnn_data_sdk::data_objects::Float64Value floatValue(cfg.passedValue);
            auto valueOffset = builder.CreateStruct(floatValue).Union();
            tensorOffsets.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                cfg.uid,
                cfg.name.c_str(),
                cfg.dataType,
                &cfg.strides,
                &cfg.dims,
                cfg.isVirtual,
                hipdnn_data_sdk::data_objects::TensorValue::Float64Value,
                valueOffset));
        }
        else
        {
            tensorOffsets.push_back(
                hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                            cfg.uid,
                                                                            cfg.name.c_str(),
                                                                            cfg.dataType,
                                                                            &cfg.strides,
                                                                            &cfg.dims,
                                                                            cfg.isVirtual));
        }
    }

    // Create peer_stats_tensor_uid with populated values (should be rejected)
    std::vector<int64_t> peerStatsUids = {100, 101, 102};
    auto peerStatsOffset = builder.CreateVector(peerStatsUids);

    // Create BatchnormAttributes WITH peer_stats populated
    auto bnAttrs = hipdnn_data_sdk::data_objects::CreateBatchnormAttributes(
        builder,
        UIDs::X,
        UIDs::SCALE,
        UIDs::BIAS,
        UIDs::EPSILON,
        peerStatsOffset, // peer_stats populated - should be rejected
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        UIDs::Y,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt);

    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_training",
        DT::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnAttrs.Union()));

    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder, "test_graph", DT::FLOAT, DT::HALF, DT::BFLOAT16, &tensorOffsets, &nodes);

    builder.Finish(graphOffset);

    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should throw because peer_stats is populated
    EXPECT_THROW(
        { checkBatchnormTensorConfigSupported(*attrs, graph.getTensorMap()); },
        hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormApplicabilityChecks, RejectsBatchnormBackwardWithPeerStats)
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;
    using UIDs = BnBackwardTensorIds;

    // Create valid tensor configs
    const auto& dims = shapes::INFERENCE_4D[0];
    auto configs = createBatchnormBackwardTensors(
        bn_type_configs::ALL_FLOAT, dims, TensorLayout::NCHW, false, false);

    flatbuffers::FlatBufferBuilder builder;

    // Build tensor attributes
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>> tensorOffsets;
    tensorOffsets.reserve(configs.size());
    for(const auto& cfg : configs)
    {
        tensorOffsets.push_back(
            hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                        cfg.uid,
                                                                        cfg.name.c_str(),
                                                                        cfg.dataType,
                                                                        &cfg.strides,
                                                                        &cfg.dims,
                                                                        cfg.isVirtual));
    }

    // Create peer_stats_tensor_uid with populated values (should be rejected)
    std::vector<int64_t> peerStatsUids = {200, 201};
    auto peerStatsOffset = builder.CreateVector(peerStatsUids);

    // Create BatchnormBackwardAttributes WITH peer_stats populated
    auto bnBwdAttrs = hipdnn_data_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        UIDs::DY,
        UIDs::X,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        UIDs::SCALE,
        peerStatsOffset, // peer_stats populated - should be rejected
        UIDs::DX,
        UIDs::DSCALE,
        UIDs::DBIAS);

    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_backward",
        DT::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttrs.Union()));

    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder, "test_graph", DT::FLOAT, DT::HALF, DT::BFLOAT16, &tensorOffsets, &nodes);

    builder.Finish(graphOffset);

    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should throw because peer_stats is populated
    EXPECT_THROW(
        { checkBatchnormTensorConfigSupported(*attrs, graph.getTensorMap()); },
        hipdnn_plugin_sdk::HipdnnPluginException);
}
