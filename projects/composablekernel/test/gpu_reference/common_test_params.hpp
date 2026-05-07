// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include <vector>

namespace ck {
namespace test {

// Common test shapes for all convolution tests (fwd, bwd_data, bwd_weight)
namespace conv_test_shapes {

// 2D Conv, FP16, Small
inline ck::utils::conv::ConvParam get_2d_small()
{
    return ck::utils::conv::ConvParam(2,      // num_dim_spatial
                                      1,      // G
                                      2,      // N
                                      8,      // K
                                      8,      // C
                                      {3, 3}, // filter_spatial
                                      {7, 7}, // input_spatial
                                      {1, 1}, // strides
                                      {1, 1}, // dilations
                                      {0, 0}, // left_pads
                                      {0, 0}  // right_pads
    );
}

// 2D Conv, FP32, Medium
inline ck::utils::conv::ConvParam get_2d_medium()
{
    return ck::utils::conv::ConvParam(2,        // num_dim_spatial
                                      1,        // G
                                      4,        // N
                                      16,       // K
                                      16,       // C
                                      {3, 3},   // filter_spatial
                                      {14, 14}, // input_spatial
                                      {1, 1},   // strides
                                      {1, 1},   // dilations
                                      {0, 0},   // left_pads
                                      {0, 0}    // right_pads
    );
}

// 1D Conv, FP16
inline ck::utils::conv::ConvParam get_1d()
{
    return ck::utils::conv::ConvParam(1,    // num_dim_spatial
                                      1,    // G
                                      2,    // N
                                      8,    // K
                                      8,    // C
                                      {3},  // filter_spatial
                                      {16}, // input_spatial
                                      {1},  // strides
                                      {1},  // dilations
                                      {0},  // left_pads
                                      {0}   // right_pads
    );
}

// 3D Conv, FP16, Small
inline ck::utils::conv::ConvParam get_3d_small()
{
    return ck::utils::conv::ConvParam(3,         // num_dim_spatial
                                      1,         // G
                                      1,         // N
                                      8,         // K
                                      8,         // C
                                      {3, 3, 3}, // filter_spatial
                                      {5, 5, 5}, // input_spatial
                                      {1, 1, 1}, // strides
                                      {1, 1, 1}, // dilations
                                      {0, 0, 0}, // left_pads
                                      {0, 0, 0}  // right_pads
    );
}

// 2D Conv with stride
inline ck::utils::conv::ConvParam get_2d_stride2()
{
    return ck::utils::conv::ConvParam(2,      // num_dim_spatial
                                      1,      // G
                                      2,      // N
                                      8,      // K
                                      8,      // C
                                      {3, 3}, // filter_spatial
                                      {8, 8}, // input_spatial
                                      {2, 2}, // strides
                                      {1, 1}, // dilations
                                      {0, 0}, // left_pads
                                      {0, 0}  // right_pads
    );
}

// 2D Grouped Conv, FP16, G=2
inline ck::utils::conv::ConvParam get_2d_grouped_g2()
{
    return ck::utils::conv::ConvParam(2,      // num_dim_spatial
                                      2,      // G
                                      2,      // N
                                      8,      // K (8 total output channels)
                                      16,     // C (16 total input channels, 8 per group with G=2)
                                      {3, 3}, // filter_spatial
                                      {7, 7}, // input_spatial
                                      {1, 1}, // strides
                                      {1, 1}, // dilations
                                      {0, 0}, // left_pads
                                      {0, 0}  // right_pads
    );
}

// 2D Grouped Conv, FP32, G=4
inline ck::utils::conv::ConvParam get_2d_grouped_g4()
{
    return ck::utils::conv::ConvParam(2,      // num_dim_spatial
                                      4,      // G
                                      1,      // N
                                      16,     // K (16 total output channels)
                                      16,     // C (16 total input channels, 4 per group with G=4)
                                      {3, 3}, // filter_spatial
                                      {8, 8}, // input_spatial
                                      {1, 1}, // strides
                                      {1, 1}, // dilations
                                      {0, 0}, // left_pads
                                      {0, 0}  // right_pads
    );
}

} // namespace conv_test_shapes
} // namespace test
} // namespace ck
