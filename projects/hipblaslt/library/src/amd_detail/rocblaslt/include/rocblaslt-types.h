/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/*! \file
 * \brief rocblaslt-types.h defines data types used by rocblaslt
 */

#pragma once
#ifndef _ROCBLASLT_TYPES_H_
#define _ROCBLASLT_TYPES_H_

#include <hip/hip_bfloat16.h>
#ifndef LEGACY_HIPBLAS_DIRECT
#include <hipblas-common/hipblas-common.h>
#else
#include <hipblas/hipblas.h>
#endif
#include <hipblaslt/hipblaslt.h>
#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <vector>

#include <hipblaslt/hipblaslt-ext.hpp>

#define ROCBLASLT_KERNEL __global__
#define ROCBLASLT_DEVICE_ILF __device__

/*! \ingroup types_module
 *  \brief Specifies whether int32 or int64 is used.
 */
#if defined(rocblaslt_ILP64)
typedef int64_t rocblaslt_int;
#else
typedef int32_t rocblaslt_int;
#endif

/* Forward declaration of hipStream_t */
typedef struct ihipStream_t* hipStream_t;

/*! \ingroup types_module
 *  \brief Handle to the rocBLASLt library context queue.
 *
 *  \details
 *  The rocBLASLt handle is a structure holding the rocBLASLt library context.
 * It must be initialized using \ref rocblaslt_create_handle and the returned
 * handle must be passed to all subsequent library function calls. It should be
 * destroyed at the end using \ref rocblaslt_destroy_handle.
 */
typedef struct _rocblaslt_handle* rocblaslt_handle;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix.
 *
 *  \details
 *  The rocBLASLt matrix descriptor is a structure holding all properties of a
 * matrix. It must be initialized using \ref rocblaslt_create_mat_descr and the
 * returned descriptor must be passed to all subsequent library calls that
 * involve the matrix. It should be destroyed at the end using \ref
 * rocblaslt_destroy_mat_descr.
 */
typedef struct _rocblaslt_matrix_layout* rocblaslt_matrix_layout;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix multiplication operation
 *
 *  \details
 *  The rocBLASLt matrix multiplication descriptor is a structure holding
 *  the description of the matrix multiplication operation.
 *  It is initialized with \ref rocblaslt_matmul_desc_create function.
 */
typedef struct _rocblaslt_matmul_desc* rocblaslt_matmul_desc;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix multiplication preference.
 *
 *  \details
 *  It is initialized with \ref rocblaslt_matmul_preference_create function.
 */
typedef struct _rocblaslt_matmul_preference* rocblaslt_matmul_preference;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix multiplication algorithm.
 *
 *  \details
 *  It is initialized with \ref rocblaslt_matmul_algo_get_heuristic function.
 */
// typedef struct _rocblaslt_matmul_algo rocblaslt_matmul_algo;

/*! \ingroup types_module
 *  \brief Descriptor of the heristic result.
 *
 *  \details
 *  It is initialized with \ref rocblaslt_matmul_algo_get_heuristic function.
 */
// typedef struct _rocblaslt_matmul_heuristic_result
// rocblaslt_matmul_heuristic_result;
#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Single precision floating point type */
typedef float rocblaslt_float;

typedef hip_bfloat16 rocblaslt_bfloat16;

#ifdef ROCM_USE_FLOAT16
typedef _Float16 rocblaslt_half;
#else
/*! \brief Structure definition for rocblaslt_half */
typedef struct rocblaslt_half
{
    uint16_t data;
} rocblaslt_half;
#endif

typedef struct
{
    uint8_t data;
} rocblaslt_f8_fnuz;

typedef struct
{
    uint8_t data;
} rocblaslt_bf8_fnuz;

typedef struct
{
    uint8_t data;
} rocblaslt_f8;

typedef struct
{
    uint8_t data;
} rocblaslt_bf8;

typedef int8_t  rocblasltInt8;
typedef int32_t rocblasltInt32;

/*! \ingroup types_module
 *  \brief Specify the postprocessing options for the epilogue
 *
 *  \details
 *  The \ref rocblaslt_epilogue indicates the postprocessing for the epilogue.
 */
typedef enum rocblaslt_epilogue_
{
    ROCBLASLT_EPILOGUE_DEFAULT            = 1,
    ROCBLASLT_EPILOGUE_RELU               = 2,
    ROCBLASLT_EPILOGUE_BIAS               = 4,
    ROCBLASLT_EPILOGUE_RELU_BIAS          = 6,
    ROCBLASLT_EPILOGUE_GELU               = 32,
    ROCBLASLT_EPILOGUE_GELU_BIAS          = 36,
    ROCBLASLT_EPILOGUE_RELU_AUX           = 130,
    ROCBLASLT_EPILOGUE_RELU_AUX_BIAS      = 134,
    ROCBLASLT_EPILOGUE_GELU_AUX           = 160,
    ROCBLASLT_EPILOGUE_GELU_AUX_BIAS      = 164,
    ROCBLASLT_EPILOGUE_DGELU              = 192,
    ROCBLASLT_EPILOGUE_DGELU_BGRAD        = 208,
    ROCBLASLT_EPILOGUE_BGRADA             = 256,
    ROCBLASLT_EPILOGUE_BGRADB             = 512,
    ROCBLASLT_EPILOGUE_SIGMOID            = 1024,
    ROCBLASLT_EPILOGUE_SWISH_EXT          = 65536,
    ROCBLASLT_EPILOGUE_SWISH_BIAS_EXT     = 65540,
    ROCBLASLT_EPILOGUE_CLAMP_EXT          = 131072,
    ROCBLASLT_EPILOGUE_CLAMP_BIAS_EXT     = 131076,
    ROCBLASLT_EPILOGUE_CLAMP_AUX_EXT      = 131200,
    ROCBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT = 131204,
} rocblaslt_epilogue;

/*! \ingroup types_module
 *  \brief Indicates if the pointer is device pointer or host pointer.
 *
 *  \details
 *  The \ref rocblaslt_pointer_mode indicates whether scalar values are passed
 * by reference on the host or device. The \ref rocblaslt_pointer_mode can be
 * changed by rocblaslt_set_pointer_mode(). The currently used pointer mode can
 * be obtained by rocblaslt_get_pointer_mode().
 */
typedef enum rocblaslt_pointer_mode_
{
    rocblaslt_pointer_mode_host   = 0, /**< scalar pointers are in host memory. */
    rocblaslt_pointer_mode_device = 1, /**< scalar pointers are in device memory. */
    rocblaslt_pointer_mode_alpha_device_vector_beta_host
    = 4 /** alpha pointer targets a device memory vector of length equal to the number of rows of matrix D, and beta is a single value in host memory. */
} rocblaslt_pointer_mode;

/*! \ingroup types_module
 *  \brief Indicates if layer is active with bitmask.
 *
 *  \details
 *  The \ref rocblaslt_layer_mode bit mask indicates the logging
 * characteristics.
 */
typedef enum rocblaslt_layer_mode
{
    rocblaslt_layer_mode_none                 = 0, /**< layer is not active. */
    rocblaslt_layer_mode_log_error            = 1, /**< layer is in error mode. */
    rocblaslt_layer_mode_log_trace            = 2, /**< layer is in trace mode. */
    rocblaslt_layer_mode_log_hints            = 4, /**< layer is in hints mode. */
    rocblaslt_layer_mode_log_info             = 8, /**< layer is in info mode. */
    rocblaslt_layer_mode_log_api              = 16, /**< layer is in api mode. */
    rocblaslt_layer_mode_log_bench            = 32, /**< layer is in bench mode. */
    rocblaslt_layer_mode_log_profile          = 64, /**< layer is in profile mode. */
    rocblaslt_layer_mode_log_extended_profile = 128, /**< layer is in Extended profile mode. */
} rocblaslt_layer_mode;

/*! \ingroup types_module
 *  \brief Indicates if layer is active with level.
 *
 *  \details
 *  The \ref rocblaslt_layer_level number indicates the logging characteristics.
 *  A higher log level will show logs including the lower log level.
 */
typedef enum rocblaslt_layer_level
{
    rocblaslt_layer_level_none      = 0, /**< layer is not active. */
    rocblaslt_layer_level_log_error = 1, /**< layer is in error mode. */
    rocblaslt_layer_level_log_trace = 2, /**< layer is in trace mode. */
    rocblaslt_layer_level_log_hints = 3, /**< layer is in hints mode. */
    rocblaslt_layer_level_log_info  = 4, /**< layer is in info mode. */
    rocblaslt_layer_level_log_api   = 5, /**< layer is in api mode. */
} rocblaslt_layer_level;

/*! \ingroup types_module
 *  \brief List of rocblaslt status codes definition.
 *
 *  \details
 *  This is a list of the \ref rocblaslt_status types that are used by the
 * rocBLASLt library.
 */
typedef enum rocblaslt_status_
{
    rocblaslt_status_success                 = 0, /**< success. */
    rocblaslt_status_invalid_handle          = 1, /**< handle not initialized, invalid or null. */
    rocblaslt_status_not_implemented         = 2, /**< function is not implemented. */
    rocblaslt_status_invalid_pointer         = 3, /**< invalid pointer parameter. */
    rocblaslt_status_invalid_size            = 4, /**< invalid size parameter. */
    rocblaslt_status_memory_error            = 5, /**< failed memory allocation, copy, dealloc. */
    rocblaslt_status_internal_error          = 6, /**< other internal library failure. */
    rocblaslt_status_invalid_value           = 7, /**< invalid value parameter. */
    rocblaslt_status_arch_mismatch           = 8, /**< device arch is not supported. */
    rocblaslt_status_zero_pivot              = 9, /**< encountered zero pivot. */
    rocblaslt_status_not_initialized         = 10, /**< descriptor has not been initialized. */
    rocblaslt_status_type_mismatch           = 11, /**< index types do not match. */
    rocblaslt_status_requires_sorted_storage = 12, /**< sorted storage required. */
    rocblaslt_status_continue                = 13 /**< nothing preventing function to proceed. */
} rocblaslt_status;

/*! \ingroup types_module
 *  \brief Specify the compute precision modes of the matrix
 *
 *  \details
 */
typedef enum rocblaslt_compute_type_
{
    rocblaslt_compute_f16          = 0, /**< 16-bit floating-point precision. */
    rocblaslt_compute_f16_pedantic = 1, /**< compute will be exactly 16-bit precision */
    rocblaslt_compute_f32          = 2, /**< 32-bit floating-point precision. */
    rocblaslt_compute_f32_pedantic = 3, /**< compute will be exactly 32-bit precision */
    rocblaslt_compute_f32_fast_f16
    = 4, /**< F16 compute for 16-bit input and 32-bit output matrices */
    rocblaslt_compute_f32_fast_bf16
    = 5, /**< BF16 compute for 16-bit input and 32-bit output matrices */
    rocblaslt_compute_f32_fast_xf32 = 6, /**< XF32 compute for 32-bit input and output matrices */
    rocblaslt_compute_f64           = 7, /**< 64-bit floating-point precision. */
    rocblaslt_compute_f64_pedantic  = 8, /**< compute will be exactly 64-bit precision */
    rocblaslt_compute_i32           = 9, /**< 32-bit integer precision. */
    rocblaslt_compute_i32_pedantic  = 10, /**< compute will be exactly 32-bit integer precision */
    rocblaslt_compute_f32_fast_f8_fnuz  = 100, /**< 32-bit input can use fp8 compute */
    rocblaslt_compute_f32_fast_bf8_fnuz = 101, /**< 32-bit input can use bf8 compute */
    rocblaslt_compute_f32_fast_f8bf8_fnuz
    = 102, /**< 32-bit input can use fp8 for A and bf8 for B compute */
    rocblaslt_compute_f32_fast_bf8f8_fnuz
    = 103, /**< 32-bit input can use bf8 for A and fp8 for B compute */
    rocblaslt_compute_f32_fast_f8  = 104, /**< 32-bit input can use fp8 compute */
    rocblaslt_compute_f32_fast_bf8 = 105, /**< 32-bit input can use bf8 compute */
    rocblaslt_compute_f32_fast_f8bf8
    = 106, /**< 32-bit input can use fp8 for A and bf8 for B compute */
    rocblaslt_compute_f32_fast_bf8f8
    = 107, /**< 32-bit input can use bf8 for A and fp8 for B compute */
} rocblaslt_compute_type;

/*! \ingroup types_module
 *  \brief Specify the additional attributes of a matrix descriptor
 *
 *  \details
 *  The rocblaslt_matrix_layout_attribute is used in the
 *  \ref rocblaslt_matrix_layout_set_attribute and \ref
 * rocblaslt_matrix_layout_get_attribute functions
 */
typedef enum rocblaslt_matrix_layout_attribute_
{
    ROCBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 0, /**< number of matrices in a batch. */
    ROCBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET
    = 1, /**< stride between consecutive matrices in a batch expressed in terms
            of matrix elements. */
    ROCBLASLT_MATRIX_LAYOUT_TYPE  = 2,
    ROCBLASLT_MATRIX_LAYOUT_ORDER = 3,
    ROCBLASLT_MATRIX_LAYOUT_ROWS  = 4,
    ROCBLASLT_MATRIX_LAYOUT_COLS  = 5,
    ROCBLASLT_MATRIX_LAYOUT_LD    = 6,
    ROCBLASLT_MATRIX_LAYOUT_MAX   = 7
} rocblaslt_matrix_layout_attribute;

typedef enum
{
    /** Column-major
   *
   * Leading dimension is the stride (in elements) to the beginning of next column in memory.
   */
    ROCBLASLT_ORDER_COL = 0,
    /** Row major
   *
   * Leading dimension is the stride (in elements) to the beginning of next row in memory.
   */
    ROCBLASLT_ORDER_ROW = 1,
} rocblasLtOrder_t;

/*! \ingroup types_module
 *  \brief Specify the additional attributes of a matrix multiplication
 * descriptor
 *
 *  \details
 *  The rocblaslt_matmul_descr_attribute_ is used in the
 *  \ref rocblaslt_matmul_descr_set_attribute and \ref
 * rocblaslt_matmul_descr_get_attribute functions
 */
typedef enum rocblaslt_matmul_desc_attributes_
{
    ROCBLASLT_MATMUL_DESC_TRANSA                     = 0,
    ROCBLASLT_MATMUL_DESC_TRANSB                     = 1,
    ROCBLASLT_MATMUL_DESC_EPILOGUE                   = 2,
    ROCBLASLT_MATMUL_DESC_BIAS_POINTER               = 3,
    ROCBLASLT_MATMUL_DESC_BIAS_DATA_TYPE             = 4,
    ROCBLASLT_MATMUL_DESC_A_SCALE_POINTER            = 5,
    ROCBLASLT_MATMUL_DESC_B_SCALE_POINTER            = 6,
    ROCBLASLT_MATMUL_DESC_C_SCALE_POINTER            = 7,
    ROCBLASLT_MATMUL_DESC_D_SCALE_POINTER            = 8,
    ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = 9,
    ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER       = 10,
    ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD            = 11,
    ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE  = 12,
    ROCBLASLT_MATMUL_DESC_POINTER_MODE               = 13,
    ROCBLASLT_MATMUL_DESC_AMAX_D_POINTER             = 14,
    ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE     = 22,
    ROCBLASLT_MATMUL_DESC_A_SCALE_MODE               = 31,
    ROCBLASLT_MATMUL_DESC_B_SCALE_MODE               = 32,
    ROCBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT   = 100,
    ROCBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT,
    ROCBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT,
    ROCBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT,
    ROCBLASLT_MATMUL_DESC_MAX,
} rocblaslt_matmul_desc_attributes;

/*! \ingroup types_module
 *  \brief Specify the matrix multiplication preference attributes.
 *
 *  \details
 *  The \ref rocblaslt_matmul_preference_attributes is used in the
 *  \ref rocblaslt_matmul_preference_get_attribute and \ref
 * rocblaslt_matmul_preference_set_attribute functions.
 */
typedef enum rocblaslt_matmul_preference_attributes_
{
    ROCBLASLT_MATMUL_PREF_SEARCH_MODE         = 0,
    ROCBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
    ROCBLASLT_MATMUL_PREF_MAX                 = 2
} rocblaslt_matmul_preference_attributes;

/********************************************************************************
 * \brief rocblaslt_matmul_algo holds the description of the matrix
 * multiplication algorithm.
typedef struct __attribute__((packed, aligned(8))) _rocblaslt_matmul_algo
{
    uint8_t data[8]             = {0};
    bool    fallback            = false;
    size_t  max_workspace_bytes = 0;
} rocblaslt_matmul_algo;
 *******************************************************************************/

/********************************************************************************
 * \brief rocblaslt_matmul_algo holds the description of the matrix
 * multiplication algorithm.
 *******************************************************************************/
typedef struct _rocblaslt_matmul_algo
{
#ifdef __cplusplus
    uint8_t data[8]             = {0}; // must match hipblasLtMatmulAlgo_t layout
    bool    fallback            = false; //
    uint8_t data_pad[7]         = {0}; // has uint8_t data[16]
    size_t  max_workspace_bytes = 0;
#else
    uint8_t data[8];
    bool    fallback;
    uint8_t data_pad[7];
    size_t  max_workspace_bytes;
#endif
} rocblaslt_matmul_algo;

static_assert(sizeof(rocblaslt_matmul_algo) == sizeof(hipblasLtMatmulAlgo_t),
              "rocblaslt_matmul_algo struct does not match size of hipblasLtMatmulAlgo_t");

/********************************************************************************
 * \brief rocblaslt_matmul_heuristic holds the configured matrix
 * multiplication algorithm descriptor and its runtime properties.
 *******************************************************************************/
typedef struct _rocblaslt_matmul_heuristic_result
{
    rocblaslt_matmul_algo algo;
    size_t                workspaceSize = 0;
    rocblaslt_status      state         = rocblaslt_status_success;
    float                 wavesCount    = 1.0;
    int                   reserved[4];
} rocblaslt_matmul_heuristic_result;

typedef struct _rocblaslt_solutions
{
    rocblaslt_matmul_heuristic_result* heuristicResults;
    int                                algoCount;
} rocblaslt_solutions;

typedef struct _rocblaslt_matrix_transform_desc
{
    hipDataType            scaleType;
    hipblasLtPointerMode_t pointerMode{HIPBLASLT_POINTER_MODE_HOST};
    hipblasOperation_t     opA{HIPBLAS_OP_N};
    hipblasOperation_t     opB{HIPBLAS_OP_N};
} rocblaslt_matrix_transform_desc;
#ifdef __cplusplus
}
#endif

/********************************************************************
 * RocblasltContractionProblem captures the arguments for a GEMM-like *
 * contraction problem, to be passed to runContractionProblem.      *
 ********************************************************************/
struct RocblasltContractionProblem
{
    enum class ScalingFormat
    {
        None = 0,
        Scalar,
        Vector,
        Block_32_UE8M0,
        Block_32_UE8M0_32_8_EXT,
    };

    hipblasOperation_t trans_a;
    hipblasOperation_t trans_b;

    // The RocblasltContractionProblem data members should exactly match
    // Tensile's parameter types, even if rocBLAS uses differently
    // sized or signed types. The constructors should convert rocBLAS
    // types into the corresponding Tensile types stored in this class.
    size_t m;
    size_t n;
    size_t k;

    const void* alpha;

    hipDataType        a_type;
    const void*        A;
    const void* const* batch_A;
    size_t             row_stride_a;
    size_t             col_stride_a;
    size_t             batch_stride_a;

    hipDataType        b_type;
    const void*        B;
    const void* const* batch_B;
    size_t             row_stride_b;
    size_t             col_stride_b;
    size_t             batch_stride_b;

    const void* beta;

    hipDataType        c_type;
    const void*        C;
    const void* const* batch_C;
    size_t             row_stride_c;
    size_t             col_stride_c;
    size_t             batch_stride_c;

    hipDataType  d_type;
    void*        D;
    void* const* batch_D;
    size_t       row_stride_d;
    size_t       col_stride_d;
    size_t       batch_stride_d;

    void*        E;
    void* const* batch_E;
    size_t       row_stride_e;
    size_t       col_stride_e;
    size_t       batch_stride_e;

    size_t batch_count;
    bool   strided_batch;
    bool   grouped_gemm;
    bool   gradient;

    rocblaslt_compute_type compute_type;
    hipDataType            scale_type;

    const void*   bias;
    const void*   scaleA;
    const void*   scaleB;
    const void*   scaleC;
    const void*   scaleD;
    const void*   scaleE;
    const void*   scaleAlphaVec;
    ScalingFormat scaleAType;
    ScalingFormat scaleBType;

    hipDataType        bias_type;
    hipDataType        aux_type;
    rocblaslt_epilogue epilogue;
    void*              amaxD;
    void*              workspace;
    size_t             workspaceSize;
    float              act0;
    float              act1;

    hipStream_t stream;
    void*       Synchronizer;
    bool        swizzleA;
    bool        swizzleB;

    // gemm_ex
    // gemm_strided_batched_ex
    RocblasltContractionProblem(hipblasOperation_t     trans_a,
                                hipblasOperation_t     trans_b,
                                int64_t                m,
                                int64_t                n,
                                int64_t                k,
                                const void*            alpha,
                                hipDataType            a_type,
                                const void*            A,
                                const void* const*     batch_A,
                                int64_t                ld_a,
                                int64_t                batch_stride_a,
                                hipDataType            b_type,
                                const void*            B,
                                const void* const*     batch_B,
                                int64_t                ld_b,
                                int64_t                batch_stride_b,
                                const void*            beta,
                                hipDataType            c_type,
                                const void*            C,
                                const void* const*     batch_C,
                                int64_t                ld_c,
                                int64_t                batch_stride_c,
                                hipDataType            d_type,
                                void*                  D,
                                void* const*           batch_D,
                                int64_t                ld_d,
                                int64_t                batch_stride_d,
                                void*                  E,
                                void* const*           batch_E,
                                int64_t                ld_e,
                                int64_t                batch_stride_e,
                                int64_t                batch_count,
                                bool                   strided_batch,
                                bool                   grouped_gemm,
                                bool                   gradient,
                                rocblaslt_compute_type compute_type,
                                hipDataType            scale_type,
                                const void*            bias,
                                const void*            scaleA,
                                const void*            scaleB,
                                const void*            scaleC,
                                const void*            scaleD,
                                const void*            scaleE,
                                const void*            scaleAlphaVec,
                                ScalingFormat          scaleAType,
                                ScalingFormat          scaleBType,
                                hipDataType            bias_type,
                                hipDataType            aux_type,
                                rocblaslt_epilogue     epilogue,
                                void*                  amaxD,
                                void*                  workspace,
                                size_t                 workspaceSize,
                                float                  act0,
                                float                  act1,
                                hipStream_t            stream,
                                void*                  Synchronizer,
                                bool                   swizzleA,
                                bool                   swizzleB);
};

namespace rocblaslt
{

    enum class RocGemmType
    {
        ROCBLASLT_GEMM             = 1,
        ROCBLASLT_GROUPED_GEMM     = 2,
        ROCBLASLT_GEMMTYPE_UNKNOWN = 3
    };

    class RocGemmProblemTypeV2
    {
    public:
        hipblasOperation_t     op_a;
        hipblasOperation_t     op_b;
        hipDataType            type_a;
        hipDataType            type_b;
        hipDataType            type_c;
        hipDataType            type_d;
        rocblaslt_compute_type type_compute;
        hipblasLtOrder_t       order_a;
        hipblasLtOrder_t       order_b;
    };

    class RocGemmEpilogueV2
    {
    public:
        rocblaslt_epilogue                         mode           = ROCBLASLT_EPILOGUE_DEFAULT;
        hipDataType                                bias_data_type = HIPBLASLT_DATATYPE_INVALID;
        hipDataType                                aux_data_type  = HIPBLASLT_DATATYPE_INVALID;
        int                                        aux_ld         = 0;
        int                                        aux_stride     = 0;
        RocblasltContractionProblem::ScalingFormat scaling_a_type
            = RocblasltContractionProblem::ScalingFormat::None;
        RocblasltContractionProblem::ScalingFormat scaling_b_type
            = RocblasltContractionProblem::ScalingFormat::None;
        float act0 = 0.f;
        float act1 = 0.f;
    };

    class RocTuningV2
    {
    public:
        uint16_t gsu = 0;
        int16_t  wgm = 0;
    };

    struct RocGemmInputsV2
    {
        void* a     = nullptr;
        void* b     = nullptr;
        void* c     = nullptr;
        void* d     = nullptr;
        void* alpha = nullptr;
        void* beta  = nullptr;
        // Epilogue inputs
        void* bias          = nullptr;
        void* scaleA        = nullptr;
        void* scaleB        = nullptr;
        void* scaleC        = nullptr;
        void* scaleD        = nullptr;
        void* scaleE        = nullptr;
        void* scaleAlphaVec = nullptr;
        void* aux           = nullptr;
        void* amaxD         = nullptr;
    };
} // End of namespace rocblaslt

#endif /* _ROCBLASLT_TYPES_H_ */
