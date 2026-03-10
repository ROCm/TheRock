// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.

#ifndef SIMDOJO_SIM_TYPES_H_
#define SIMDOJO_SIM_TYPES_H_

#include <cstdint>
#include <limits>

namespace simdojo {

/// @brief Simulation tick type.
using Tick = uint64_t;

/// @brief Maximum tick value (used as a sentinel for "no event").
inline constexpr Tick TICK_MAX = std::numeric_limits<Tick>::max();

/// @brief Unique identifier for a Node or Component.
using ComponentID = uint32_t;

/// @brief Unique identifier for a Link.
using LinkID = uint32_t;

/// @brief Unique identifier for a Port.
using PortID = uint32_t;

/// @brief Identifier for a simulation partition (thread affinity).
using PartitionID = uint32_t;

/// @brief Sentinel value indicating no partition has been assigned.
inline constexpr PartitionID INVALID_PARTITION_ID = std::numeric_limits<PartitionID>::max();

} // namespace simdojo

#endif // SIMDOJO_SIM_TYPES_H_
