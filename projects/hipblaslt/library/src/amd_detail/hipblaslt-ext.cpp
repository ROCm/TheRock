/*******************************************************************************
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "hipblaslt/hipblaslt-ext.hpp"
#include "exceptions.hpp"
#include "hipblaslt_internal.hpp"
#include <Debug.hpp>
#include <algorithm>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-types.h>
#include <iostream>
#include <rocblaslt.h>

namespace hipblaslt_ext
{
    class GemmPreference::GemmPreferenceImpl
    {
    public:
        size_t workspace_bytes;
    };

    GemmPreference::GemmPreference()
        : pimpl(std::make_unique<GemmPreferenceImpl>())
    {
    }

    GemmPreference::~GemmPreference() = default;

    GemmPreference::GemmPreference(const GemmPreference& pref)
        : pimpl(std::make_unique<GemmPreferenceImpl>(*pref.pimpl))
    {
    }

    GemmPreference& GemmPreference::operator=(const GemmPreference& pref)
    {
        *pimpl = *pref.pimpl;
        return *this;
    }

    GemmPreference::GemmPreference(GemmPreference&& pref)            = default;
    GemmPreference& GemmPreference::operator=(GemmPreference&& pref) = default;

    void GemmPreference::setMaxWorkspaceBytes(size_t workspaceBytes)
    {
        pimpl->workspace_bytes = workspaceBytes;
    }

    const size_t GemmPreference::getMaxWorkspaceBytes() const
    {
        return pimpl->workspace_bytes;
    }

    class GemmProblemType::GemmProblemTypeImpl
    {
    public:
        hipblasOperation_t   op_a; //!< The A martix transpose
        hipblasOperation_t   op_b; //!< The B matrix transpose
        hipDataType          type_a; //!< The A matrix datatype.
        hipDataType          type_b; //!< The B matrix datatype.
        hipDataType          type_c; //!< The C matrix datatype.
        hipDataType          type_d; //!< The D matrix datatype.
        hipblasComputeType_t type_compute; //!< The compute datatype.
        hipblasLtOrder_t     order_a; //!< The A martix data layout order
        hipblasLtOrder_t     order_b; //!< The B martix data layout order
    };

    GemmProblemType::GemmProblemType()
        : pimpl(std::make_unique<GemmProblemTypeImpl>())
    {
    }

    GemmProblemType::GemmProblemType(hipblasOperation_t   opA,
                                     hipblasOperation_t   opB,
                                     hipDataType          typeA,
                                     hipDataType          typeB,
                                     hipDataType          typeC,
                                     hipDataType          typeD,
                                     hipblasComputeType_t typeCompute)
        : pimpl(std::make_unique<GemmProblemTypeImpl>())
    {
        pimpl->op_a         = opA;
        pimpl->op_b         = opB;
        pimpl->type_a       = typeA;
        pimpl->type_b       = typeB;
        pimpl->type_c       = typeC;
        pimpl->type_d       = typeD;
        pimpl->type_compute = typeCompute;

        // default value of order is COL despite of opA/B,
        // currently only swizzle cases use the variables
        pimpl->order_a = HIPBLASLT_ORDER_COL;
        pimpl->order_b = HIPBLASLT_ORDER_COL;
    }

    GemmProblemType::~GemmProblemType() = default;

    GemmProblemType::GemmProblemType(const GemmProblemTypeImpl& impl)
        : pimpl(std::make_unique<GemmProblemTypeImpl>(impl))
    {
    }

    GemmProblemType& GemmProblemType::operator=(const GemmProblemTypeImpl& impl)
    {
        *this->pimpl = impl;
        return *this;
    }

    GemmProblemType::GemmProblemType(const GemmProblemType& type)
        : pimpl(std::make_unique<GemmProblemTypeImpl>(*type.pimpl))
    {
    }

    GemmProblemType& GemmProblemType::operator=(const GemmProblemType& type)
    {
        *pimpl = *type.pimpl;
        return *this;
    }

    GemmProblemType::GemmProblemType(GemmProblemType&& type)            = default;
    GemmProblemType& GemmProblemType::operator=(GemmProblemType&& type) = default;

    void GemmProblemType::setOpA(hipblasOperation_t op)
    {
        pimpl->op_a = op;
    }

    void GemmProblemType::setOpB(hipblasOperation_t op)
    {
        pimpl->op_b = op;
    }

    void GemmProblemType::setTypeA(hipDataType type)
    {
        pimpl->type_a = type;
    }

    void GemmProblemType::setTypeB(hipDataType type)
    {
        pimpl->type_b = type;
    }

    void GemmProblemType::setTypeC(hipDataType type)
    {
        pimpl->type_c = type;
    }

    void GemmProblemType::setTypeD(hipDataType type)
    {
        pimpl->type_d = type;
    }

    void GemmProblemType::setTypeCompute(hipblasComputeType_t type)
    {
        pimpl->type_compute = type;
    }

    void GemmProblemType::setOrderA(hipblasLtOrder_t order)
    {
        pimpl->order_a = order;
    }

    void GemmProblemType::setOrderB(hipblasLtOrder_t order)
    {
        pimpl->order_b = order;
    }

    hipblasOperation_t GemmProblemType::getOpA() const
    {
        return pimpl->op_a;
    }

    hipblasOperation_t GemmProblemType::getOpB() const
    {
        return pimpl->op_b;
    }

    hipDataType GemmProblemType::getTypeA() const
    {
        return pimpl->type_a;
    }

    hipDataType GemmProblemType::getTypeB() const
    {
        return pimpl->type_b;
    }

    hipDataType GemmProblemType::getTypeC() const
    {
        return pimpl->type_c;
    }

    hipDataType GemmProblemType::getTypeD() const
    {
        return pimpl->type_d;
    }

    hipblasComputeType_t GemmProblemType::getTypeCompute() const
    {
        return pimpl->type_compute;
    }

    hipblasLtOrder_t GemmProblemType::getOrderA() const
    {
        return pimpl->order_a;
    }

    hipblasLtOrder_t GemmProblemType::getOrderB() const
    {
        return pimpl->order_b;
    }

    class GemmEpilogue::GemmEpilogueImpl
    {
    public:
        hipblasLtEpilogue_t                        mode           = HIPBLASLT_EPILOGUE_DEFAULT;
        hipDataType                                bias_data_type = HIPBLASLT_DATATYPE_INVALID;
        hipDataType                                aux_data_type  = HIPBLASLT_DATATYPE_INVALID;
        int                                        aux_ld         = 0;
        int                                        aux_stride     = 0;
        RocblasltContractionProblem::ScalingFormat scaling_a_type
            = RocblasltContractionProblem::ScalingFormat::None;
        RocblasltContractionProblem::ScalingFormat scaling_b_type
            = RocblasltContractionProblem::ScalingFormat::None;
        float act0;
        float act1;
    };

    GemmEpilogue::GemmEpilogue()
        : pimpl(std::make_unique<GemmEpilogueImpl>())
    {
    }

    GemmEpilogue::~GemmEpilogue() = default;

    GemmEpilogue::GemmEpilogue(const GemmEpilogue& epilogue)
        : pimpl(std::make_unique<GemmEpilogueImpl>(*epilogue.pimpl))
    {
    }

    GemmEpilogue& GemmEpilogue::operator=(const GemmEpilogue& epilogue)
    {
        *pimpl = *epilogue.pimpl;
        return *this;
    }

    GemmEpilogue::GemmEpilogue(GemmEpilogue&& epilogue)            = default;
    GemmEpilogue& GemmEpilogue::operator=(GemmEpilogue&& epilogue) = default;

    void GemmEpilogue::setMode(hipblasLtEpilogue_t mode)
    {
        pimpl->mode = mode;
    }

    void GemmEpilogue::setBiasDataType(hipDataType bias_data_type)
    {
        pimpl->bias_data_type = bias_data_type;
    }

    void GemmEpilogue::setAuxDataType(hipDataType aux_data_type)
    {
        pimpl->aux_data_type = aux_data_type;
    }

    void GemmEpilogue::setAuxLeadingDimension(int aux_ld)
    {
        pimpl->aux_ld = aux_ld;
    }

    void GemmEpilogue::setAuxBatchStride(int aux_stride)
    {
        pimpl->aux_stride = aux_stride;
    }

    void GemmEpilogue::setScalingAType(hipblasLtMatmulMatrixScale_t scaling_a_type)
    {
        switch(scaling_a_type)
        {
        case HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F:
            pimpl->scaling_a_type = RocblasltContractionProblem::ScalingFormat::Scalar;
            break;
        case HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F:
            pimpl->scaling_a_type = RocblasltContractionProblem::ScalingFormat::Vector;
            break;
        case HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3:
        case HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0:
        default:
            std::cerr << "Unsupported scaling type for A matrix: "
                      << static_cast<int>(scaling_a_type) << std::endl;
            throw std::invalid_argument("Unsupported scaling type for A matrix");
        }
    }

    void GemmEpilogue::setScalingBType(hipblasLtMatmulMatrixScale_t scaling_b_type)
    {
        switch(scaling_b_type)
        {
        case HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F:
            pimpl->scaling_b_type = RocblasltContractionProblem::ScalingFormat::Scalar;
            break;
        case HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F:
            pimpl->scaling_b_type = RocblasltContractionProblem::ScalingFormat::Vector;
            break;
        case HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3:
        case HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0:
        default:
            std::cerr << "Unsupported scaling type for B matrix: "
                      << static_cast<int>(scaling_b_type) << std::endl;
            throw std::invalid_argument("Unsupported scaling type for B matrix");
        }
    }

    void GemmEpilogue::setAct0(float act0)
    {
        pimpl->act0 = act0;
    }

    void GemmEpilogue::setAct1(float act1)
    {
        pimpl->act1 = act1;
    }

    hipblasLtEpilogue_t GemmEpilogue::getMode() const
    {
        return pimpl->mode;
    }

    hipDataType GemmEpilogue::getBiasDataType() const
    {
        return pimpl->bias_data_type;
    }

    hipDataType GemmEpilogue::getAuxDataType() const
    {
        return pimpl->aux_data_type;
    }

    int GemmEpilogue::getAuxLeadingDimension() const
    {
        return pimpl->aux_ld;
    }

    int GemmEpilogue::getAuxBatchStride() const
    {
        return pimpl->aux_stride;
    }

    hipblasLtMatmulMatrixScale_t GemmEpilogue::getScalingAType() const
    {
        switch(pimpl->scaling_a_type)
        {
        case RocblasltContractionProblem::ScalingFormat::Scalar:
            return HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        case RocblasltContractionProblem::ScalingFormat::Vector:
            return HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        default:
            std::cerr << "Unsupported scaling type for A matrix: "
                      << static_cast<int>(pimpl->scaling_a_type) << std::endl;
            throw std::invalid_argument("Unsupported scaling type for A matrix");
        }
    }

    hipblasLtMatmulMatrixScale_t GemmEpilogue::getScalingBType() const
    {
        switch(pimpl->scaling_b_type)
        {
        case RocblasltContractionProblem::ScalingFormat::Scalar:
            return HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        case RocblasltContractionProblem::ScalingFormat::Vector:
            return HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        default:
            std::cerr << "Unsupported scaling type for B matrix: "
                      << static_cast<int>(pimpl->scaling_b_type) << std::endl;
            throw std::invalid_argument("Unsupported scaling type for B matrix");
        }
    }

    float GemmEpilogue::getAct0()
    {
        return pimpl->act0;
    }

    float GemmEpilogue::getAct1()
    {
        return pimpl->act1;
    }

    class GemmTuning::GemmTuningImpl
    {
    public:
        uint16_t splitK = 0;
        int16_t  wgm    = 0;
    };

    GemmTuning::GemmTuning()
        : pimpl(std::make_unique<GemmTuningImpl>())
    {
    }

    GemmTuning::~GemmTuning() = default;

    GemmTuning::GemmTuning(const GemmTuning& tuning)
        : pimpl(std::make_unique<GemmTuningImpl>(*tuning.pimpl))
    {
    }

    GemmTuning& GemmTuning::operator=(const GemmTuning& tuning)
    {
        *pimpl = *tuning.pimpl;
        return *this;
    }

    GemmTuning::GemmTuning(GemmTuning&& tuning)            = default;
    GemmTuning& GemmTuning::operator=(GemmTuning&& tuning) = default;

    void GemmTuning::setSplitK(uint16_t splitK)
    {
        pimpl->splitK = splitK;
    }

    void GemmTuning::setWgm(int16_t wgm)
    {
        pimpl->wgm = wgm;
    }

    uint16_t GemmTuning::getSplitK() const
    {
        return pimpl->splitK;
    }

    int16_t GemmTuning::getWgm() const
    {
        return pimpl->wgm;
    }

    class GemmInputs::GemmInputsImpl
    {
    public:
        const void* a     = nullptr;
        const void* b     = nullptr;
        const void* c     = nullptr;
        const void* d     = nullptr;
        const void* alpha = nullptr;
        const void* beta  = nullptr;
        // Epilogue inputs
        const void* bias          = nullptr;
        const void* scaleA        = nullptr;
        const void* scaleB        = nullptr;
        const void* scaleC        = nullptr;
        const void* scaleD        = nullptr;
        const void* scaleAux      = nullptr;
        const void* scaleAlphaVec = nullptr;
        const void* aux           = nullptr;
        const void* amaxD         = nullptr;
    };

    GemmInputs::GemmInputs()
        : pimpl(std::make_unique<GemmInputsImpl>())
    {
    }

    GemmInputs::~GemmInputs() = default;

    GemmInputs::GemmInputs(const GemmInputs& input)
        : pimpl(std::make_unique<GemmInputsImpl>(*input.pimpl))
    {
    }

    GemmInputs& GemmInputs::operator=(const GemmInputs& input)
    {
        *pimpl = *input.pimpl;
        return *this;
    }

    GemmInputs::GemmInputs(GemmInputs&& input)            = default;
    GemmInputs& GemmInputs::operator=(GemmInputs&& input) = default;

    void GemmInputs::setA(const void* a)
    {
        pimpl->a = a;
    }

    void GemmInputs::setB(const void* b)
    {
        pimpl->b = b;
    }

    void GemmInputs::setC(const void* c)
    {
        pimpl->c = c;
    }

    void GemmInputs::setD(const void* d)
    {
        pimpl->d = d;
    }

    void GemmInputs::setAlpha(const void* alpha)
    {
        pimpl->alpha = alpha;
    }

    void GemmInputs::setBeta(const void* beta)
    {
        pimpl->beta = beta;
    }

    void GemmInputs::setBias(const void* bias)
    {
        pimpl->bias = bias;
    }

    void GemmInputs::setScaleA(const void* scaleA)
    {
        pimpl->scaleA = scaleA;
    }

    void GemmInputs::setScaleB(const void* scaleB)
    {
        pimpl->scaleB = scaleB;
    }

    void GemmInputs::setScaleC(const void* scaleC)
    {
        pimpl->scaleC = scaleC;
    }

    void GemmInputs::setScaleD(const void* scaleD)
    {
        pimpl->scaleD = scaleD;
    }

    void GemmInputs::setScaleAux(const void* scaleAux)
    {
        pimpl->scaleAux = scaleAux;
    }

    void GemmInputs::setScaleAlphaVec(const void* scaleAlphaVec)
    {
        pimpl->scaleAlphaVec = scaleAlphaVec;
    }

    void GemmInputs::setAux(const void* aux)
    {
        pimpl->aux = aux;
    }

    void GemmInputs::setAmaxD(const void* amaxD)
    {
        pimpl->amaxD = amaxD;
    }

    const void* GemmInputs::getA() const
    {
        return pimpl->a;
    }

    const void* GemmInputs::getB() const
    {
        return pimpl->b;
    }

    const void* GemmInputs::getC() const
    {
        return pimpl->c;
    }

    const void* GemmInputs::getD() const
    {
        return pimpl->d;
    }

    const void* GemmInputs::getAlpha() const
    {
        return pimpl->alpha;
    }

    const void* GemmInputs::getBeta() const
    {
        return pimpl->beta;
    }

    const void* GemmInputs::getBias() const
    {
        return pimpl->bias;
    }

    const void* GemmInputs::getScaleA() const
    {
        return pimpl->scaleA;
    }

    const void* GemmInputs::getScaleB() const
    {
        return pimpl->scaleB;
    }

    const void* GemmInputs::getScaleC() const
    {
        return pimpl->scaleC;
    }

    const void* GemmInputs::getScaleD() const
    {
        return pimpl->scaleD;
    }

    const void* GemmInputs::getScaleAux() const
    {
        return pimpl->scaleAux;
    }

    const void* GemmInputs::getScaleAlphaVec() const
    {
        return pimpl->scaleAlphaVec;
    }

    const void* GemmInputs::getAux() const
    {
        return pimpl->aux;
    }

    const void* GemmInputs::getAmaxD() const
    {
        return pimpl->amaxD;
    }

    // End of pimpl classes
    /////////////////////////////////////////////////////

    bool currentArchSupportsFp8()
    {
        using std::begin;
        using std::end;

        static const std::string fp8Archs[] = {"gfx942", "gfx950"};
        const auto               archName   = rocblaslt_internal_get_arch_name();
        return std::find(begin(fp8Archs), end(fp8Archs), archName) != end(fp8Archs);
    }

    template <typename SrcType, typename DstType, typename ScaleType = float>
    __global__ void datatypeConversion(const SrcType*   src,
                                       DstType*         dst,
                                       const ScaleType* scale,
                                       std::size_t      numElements)
    {
        const auto tId        = threadIdx.x;
        const auto bId        = blockIdx.x;
        const auto blockSize  = blockDim.x * blockDim.y * blockDim.z;
        const auto elemOffset = bId * blockSize + tId;
        const auto scaleValue = scale ? *scale : 1.f;

        if(elemOffset < numElements)
        {
            dst[elemOffset] = DstType(float(src[elemOffset]) * scaleValue);
        }
    }

    template <typename SrcType, typename DstType>
    void datatypeConversionCpu(const SrcType* src, DstType* dst, std::size_t numElements)
    {
        for(std::size_t i = 0; i < numElements; ++i)
        {
            dst[i] = DstType(src[i]);
        }
    }

    auto NullDeleter = [](void*) { return hipSuccess; };

    HipBufferPtr makeHipBuffer(std::size_t numBytes)
    {
        if(!numBytes)
        {
            return HipBufferPtr(nullptr, NullDeleter);
        }

        void* ptr = nullptr;
        auto  err = hipMalloc(&ptr, numBytes);

        if(err != hipSuccess)
        {
            return HipBufferPtr(nullptr, NullDeleter);
        }

        return HipBufferPtr(ptr, &hipFree);
    }

    ////////////////////////////////////////////////////////////
    // Gemm Instance
    ////////////////////////////////////////////////////////////

    GemmInstance::GemmInstance(hipblasLtHandle_t handle, GemmType type)
        : m_gemm_type(type)
        , m_handle(handle)
    {
    }

    GemmInstance::GemmInstance(GemmInstance&& rhs) noexcept            = default;
    GemmInstance& GemmInstance::operator=(GemmInstance&& rhs) noexcept = default;

    GemmType GemmInstance::getGemmType()
    {
        return m_gemm_type;
    }

    size_t GemmInstance::getGemmCount()
    {
        return m_gemm_count;
    }

    hipblasStatus_t GemmInstance::algoGetHeuristic(
        const int                                      requestedAlgoCount,
        const GemmPreference&                          pref,
        std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults)
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtAlgoGetHeuristicV2Cpp");
        if(m_gemm_count == 0)
        {
            rocblaslt::Debug::Instance().markerStop();
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto results
            = reinterpret_cast<std::vector<rocblaslt_matmul_heuristic_result>*>(&heuristicResults);
        results->clear();
        auto status = RocBlasLtStatusToHIPStatus(
            rocblaslt_algo_get_heuristic_cpp((rocblaslt_handle)m_handle,
                                             gemmType,
                                             m_data,
                                             pref.pimpl->workspace_bytes,
                                             requestedAlgoCount,
                                             *results));
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }

    hipblasStatus_t GemmInstance::isAlgoSupported(hipblasLtMatmulAlgo_t& algo,
                                                  size_t&                workspaceSizeInBytes)
    try
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtIsAlgoSupportedCpp");
        auto                    gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto                    rocalgo  = reinterpret_cast<rocblaslt_matmul_algo*>(&algo);
        rocblaslt::RocTuningV2* tuning   = nullptr;
        auto                    status = RocBlasLtStatusToHIPStatus(rocblaslt_is_algo_supported_cpp(
            (rocblaslt_handle)m_handle, gemmType, m_data, *rocalgo, tuning, workspaceSizeInBytes));
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    hipblasStatus_t GemmInstance::isAlgoSupported(hipblasLtMatmulAlgo_t& algo,
                                                  GemmTuning&            tuning,
                                                  size_t&                workspaceSizeInBytes)
    try
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtIsAlgoSupportedTuningV2Cpp");
        auto gemmType  = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto rocalgo   = reinterpret_cast<rocblaslt_matmul_algo*>(&algo);
        auto roctuning = reinterpret_cast<rocblaslt::RocTuningV2*>(tuning.pimpl.get());
        auto status
            = RocBlasLtStatusToHIPStatus(rocblaslt_is_algo_supported_cpp((rocblaslt_handle)m_handle,
                                                                         gemmType,
                                                                         m_data,
                                                                         *rocalgo,
                                                                         roctuning,
                                                                         workspaceSizeInBytes));
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    void GemmInstance::setMaxWorkspaceBytes(size_t workspaceBytes)
    {
        m_workspace_bytes = workspaceBytes;
    }

    const size_t GemmInstance::getMaxWorkspaceBytes() const
    {
        return m_workspace_bytes;
    }

    hipblasStatus_t GemmInstance::initialize(const hipblasLtMatmulAlgo_t& algo,
                                             void*                        workspace,
                                             bool                         useUserArgs,
                                             hipStream_t                  stream)
    try
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtInitializeCpp");
        if((m_gemm_count == 0) || (workspace == nullptr && m_workspace_bytes > 0))
        {
            rocblaslt::Debug::Instance().markerStop();
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        auto                    gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto                    rocalgo  = reinterpret_cast<const rocblaslt_matmul_algo*>(&algo);
        rocblaslt::RocTuningV2* tuning   = nullptr;
        auto                    status
            = RocBlasLtStatusToHIPStatus(rocblaslt_makeArgument_cpp((rocblaslt_handle)m_handle,
                                                                    gemmType,
                                                                    *rocalgo,
                                                                    tuning,
                                                                    workspace,
                                                                    m_workspace_bytes,
                                                                    useUserArgs,
                                                                    stream,
                                                                    m_data));
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    hipblasStatus_t GemmInstance::initialize(const hipblasLtMatmulAlgo_t& algo,
                                             GemmTuning&                  tuning,
                                             void*                        workspace,
                                             bool                         useUserArgs,
                                             hipStream_t                  stream)
    try
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtInitializeTuningV2Cpp");
        if((m_gemm_count == 0) || (workspace == nullptr && m_workspace_bytes > 0))
        {
            rocblaslt::Debug::Instance().markerStop();
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        auto gemmType  = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto rocalgo   = reinterpret_cast<const rocblaslt_matmul_algo*>(&algo);
        auto roctuning = reinterpret_cast<const rocblaslt::RocTuningV2*>(tuning.pimpl.get());
        auto status
            = RocBlasLtStatusToHIPStatus(rocblaslt_makeArgument_cpp((rocblaslt_handle)m_handle,
                                                                    gemmType,
                                                                    *rocalgo,
                                                                    roctuning,
                                                                    workspace,
                                                                    m_workspace_bytes,
                                                                    useUserArgs,
                                                                    stream,
                                                                    m_data));
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    hipblasStatus_t GemmInstance::run(hipStream_t stream, hipEvent_t start, hipEvent_t stop)
    try
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtRunCpp");
        if(m_gemm_count == 0)
        {
            rocblaslt::Debug::Instance().markerStop();
            return HIPBLAS_STATUS_INVALID_VALUE;
        }

        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto status   = RocBlasLtStatusToHIPStatus(
            rocblaslt_run_cpp((rocblaslt_handle)m_handle, gemmType, m_data, stream, start, stop));
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    std::string GemmInstance::getSolutionName()
    {
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        return rocblaslt_get_solution_name_from_data_cpp(
            (rocblaslt_handle)m_handle, gemmType, m_data);
    }

    std::string GemmInstance::getKernelName()
    {
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        return rocblaslt_get_kernel_name_from_data_cpp(
            (rocblaslt_handle)m_handle, gemmType, m_data);
    }

    Gemm::Gemm(hipblasLtHandle_t    handle,
               hipblasOperation_t   opA,
               hipblasOperation_t   opB,
               hipDataType          typeA,
               hipDataType          typeB,
               hipDataType          typeC,
               hipDataType          typeD,
               hipblasComputeType_t typeCompute)
        : GemmInstance(handle, GemmType::HIPBLASLT_GEMM)
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtCreateGemmCpp");
        m_problem_types.push_back({opA, opB, typeA, typeB, typeC, typeD, typeCompute});
        rocblaslt_init_gemmData((rocblaslt_handle)m_handle,
                                static_cast<rocblaslt::RocGemmType>(m_gemm_type),
                                opA,
                                opB,
                                typeA,
                                typeB,
                                typeC,
                                typeD,
                                (rocblaslt_compute_type)typeCompute,
                                0,
                                m_data);
        rocblaslt::Debug::Instance().markerStop();
    }

    Gemm::Gemm(hipblasLtHandle_t       handle,
               hipblasLtMatmulDesc_t   matmul_descr,
               const void*             alpha,
               const void*             A,
               hipblasLtMatrixLayout_t matA,
               const void*             B,
               hipblasLtMatrixLayout_t matB,
               const void*             beta,
               const void*             C,
               hipblasLtMatrixLayout_t matC,
               void*                   D,
               hipblasLtMatrixLayout_t matD)
        : GemmInstance(handle, GemmType::HIPBLASLT_GEMM)
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtCreateGemmCAPICpp");
        auto status = setProblem(matmul_descr, alpha, A, matA, B, matB, beta, C, matC, D, matD);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            std::cout << "Failed to create instance " << status << std::endl;
        }
        rocblaslt::Debug::Instance().markerStop();
    }

    Gemm::Gemm(Gemm&&) noexcept            = default;
    Gemm& Gemm::operator=(Gemm&&) noexcept = default;

    hipblasStatus_t Gemm::setProblem(int64_t       m,
                                     int64_t       n,
                                     int64_t       k,
                                     int64_t       batch_count,
                                     GemmEpilogue& epilogue,
                                     GemmInputs&   inputs)
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtGemmSetProblemV2Cpp");
        if(n == 0 || m == 0)
        {
            rocblaslt::Debug::Instance().markerStop();
            return HIPBLAS_STATUS_INVALID_VALUE;
        }

        int64_t lda     = m_problem_types[0].getOpA() == HIPBLAS_OP_N ? m : k;
        int64_t ldb     = m_problem_types[0].getOpB() == HIPBLAS_OP_N ? k : n;
        int64_t ldc     = m;
        int64_t strideA = m * k;
        int64_t strideB = n * k;
        int64_t strideC = m * n;
        auto    status  = setProblem(m,
                                 n,
                                 k,
                                 batch_count,
                                 lda,
                                 ldb,
                                 ldc,
                                 ldc,
                                 strideA,
                                 strideB,
                                 strideC,
                                 strideC,
                                 epilogue,
                                 inputs,
                                 m_problem_types[0]);
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }

    hipblasStatus_t Gemm::setProblem(int64_t          m,
                                     int64_t          n,
                                     int64_t          k,
                                     int64_t          batch_count,
                                     int64_t          lda,
                                     int64_t          ldb,
                                     int64_t          ldc,
                                     int64_t          ldd,
                                     int64_t          strideA,
                                     int64_t          strideB,
                                     int64_t          strideC,
                                     int64_t          strideD,
                                     GemmEpilogue&    epilogue,
                                     GemmInputs&      inputs,
                                     GemmProblemType& problemtype)
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtGemmSetProblemFullV2Cpp");
        auto rocepilogue = reinterpret_cast<rocblaslt::RocGemmEpilogueV2*>(epilogue.pimpl.get());
        auto rocepinputs = reinterpret_cast<rocblaslt::RocGemmInputsV2*>(inputs.pimpl.get());
        auto rocproblemtype
            = reinterpret_cast<rocblaslt::RocGemmProblemTypeV2*>(problemtype.pimpl.get());
        auto status
            = RocBlasLtStatusToHIPStatus(rocblaslt_gemm_create_cpp((rocblaslt_handle)m_handle,
                                                                   m,
                                                                   n,
                                                                   batch_count,
                                                                   k,
                                                                   lda,
                                                                   ldb,
                                                                   ldc,
                                                                   ldd,
                                                                   strideA,
                                                                   strideB,
                                                                   strideC,
                                                                   strideD,
                                                                   *rocepilogue,
                                                                   *rocepinputs,
                                                                   *rocproblemtype,
                                                                   m_data,
                                                                   m_gemm_count));
        if(status == HIPBLAS_STATUS_SUCCESS)
        {
            m_problem_types[0] = problemtype;
        }
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }

    hipblasStatus_t Gemm::setProblem(hipblasLtMatmulDesc_t   matmul_descr,
                                     const void*             alpha,
                                     const void*             A,
                                     hipblasLtMatrixLayout_t matA,
                                     const void*             B,
                                     hipblasLtMatrixLayout_t matB,
                                     const void*             beta,
                                     const void*             C,
                                     hipblasLtMatrixLayout_t matC,
                                     void*                   D,
                                     hipblasLtMatrixLayout_t matD)
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtGemmSetProblemCAPICpp");
        auto rocproblemtypes
            = reinterpret_cast<std::vector<rocblaslt::RocGemmProblemTypeV2>*>(&m_problem_types);
        auto status = RocBlasLtStatusToHIPStatus(
            rocblaslt_gemm_create_cpp((rocblaslt_handle)m_handle,
                                      (rocblaslt_matmul_desc)matmul_descr,
                                      alpha,
                                      A,
                                      (rocblaslt_matrix_layout)matA,
                                      B,
                                      (rocblaslt_matrix_layout)matB,
                                      beta,
                                      C,
                                      (rocblaslt_matrix_layout)matC,
                                      D,
                                      (rocblaslt_matrix_layout)matD,
                                      (*rocproblemtypes)[0],
                                      m_data,
                                      m_gemm_count));
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }

    GemmProblemType Gemm::getProblemTypes()
    {
        return m_problem_types[0];
    }

    HIPBLASLT_EXPORT GroupedGemm::GroupedGemm(hipblasLtHandle_t    handle,
                                              hipblasOperation_t   opA,
                                              hipblasOperation_t   opB,
                                              hipDataType          typeA,
                                              hipDataType          typeB,
                                              hipDataType          typeC,
                                              hipDataType          typeD,
                                              hipblasComputeType_t typeCompute)
        : GemmInstance(handle, GemmType::HIPBLASLT_GROUPED_GEMM)
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtCreateGroupedGemmCpp");
        m_problem_types.push_back({opA, opB, typeA, typeB, typeC, typeD, typeCompute});
        rocblaslt_init_gemmData((rocblaslt_handle)m_handle,
                                static_cast<rocblaslt::RocGemmType>(m_gemm_type),
                                opA,
                                opB,
                                typeA,
                                typeB,
                                typeC,
                                typeD,
                                (rocblaslt_compute_type)typeCompute,
                                0,
                                m_data);
        rocblaslt::Debug::Instance().markerStop();
    }

    GroupedGemm::GroupedGemm(GroupedGemm&&) noexcept            = default;
    GroupedGemm& GroupedGemm::operator=(GroupedGemm&&) noexcept = default;

    HIPBLASLT_EXPORT GroupedGemm::GroupedGemm(hipblasLtHandle_t                     handle,
                                              std::vector<hipblasLtMatmulDesc_t>&   matmul_descr,
                                              std::vector<void*>&                   alpha,
                                              std::vector<void*>&                   A,
                                              std::vector<hipblasLtMatrixLayout_t>& matA,
                                              std::vector<void*>&                   B,
                                              std::vector<hipblasLtMatrixLayout_t>& matB,
                                              std::vector<void*>&                   beta,
                                              std::vector<void*>&                   C,
                                              std::vector<hipblasLtMatrixLayout_t>& matC,
                                              std::vector<void*>&                   D,
                                              std::vector<hipblasLtMatrixLayout_t>& matD)
        : GemmInstance(handle, GemmType::HIPBLASLT_GROUPED_GEMM)
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtCreateGroupedGemmCAPICpp");
        auto status = setProblem(matmul_descr, alpha, A, matA, B, matB, beta, C, matC, D, matD);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            std::cout << "Failed to create instance " << status << std::endl;
        }
        rocblaslt::Debug::Instance().markerStop();
    }

    hipblasStatus_t GroupedGemm::setProblem(std::vector<int64_t>&      m,
                                            std::vector<int64_t>&      n,
                                            std::vector<int64_t>&      k,
                                            std::vector<int64_t>&      batch_count,
                                            std::vector<GemmEpilogue>& epilogue,
                                            std::vector<GemmInputs>&   inputs)
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtGroupedGemmSetProblemV2Cpp");
        std::vector<int64_t> lda;
        std::vector<int64_t> ldb;
        std::vector<int64_t> ldc;
        std::vector<int64_t> ldd;
        std::vector<int64_t> strideA;
        std::vector<int64_t> strideB;
        std::vector<int64_t> strideC;
        std::vector<int64_t> strideD;
        for(size_t i = 0; i < m.size(); i++)
        {
            size_t iIdx = m_problem_types.size() == 1 ? 0 : i;
            lda.push_back(m_problem_types[iIdx].getOpA() == HIPBLAS_OP_N ? m[i] : k[i]);
            ldb.push_back(m_problem_types[iIdx].getOpB() == HIPBLAS_OP_N ? k[i] : n[i]);
            ldc.push_back(m[i]);
            ldd.push_back(m[i]);
            strideA.push_back(m[i] * k[i]);
            strideB.push_back(m[i] * k[i]);
            strideC.push_back(m[i] * k[i]);
            strideD.push_back(m[i] * k[i]);
        }
        auto status = setProblem(m,
                                 n,
                                 k,
                                 batch_count,
                                 lda,
                                 ldb,
                                 ldc,
                                 ldd,
                                 strideA,
                                 strideB,
                                 strideC,
                                 strideD,
                                 epilogue,
                                 inputs,
                                 m_problem_types[0]);
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }

    hipblasStatus_t GroupedGemm::setProblem(std::vector<int64_t>&      m,
                                            std::vector<int64_t>&      n,
                                            std::vector<int64_t>&      k,
                                            std::vector<int64_t>&      batch_count,
                                            std::vector<int64_t>&      lda,
                                            std::vector<int64_t>&      ldb,
                                            std::vector<int64_t>&      ldc,
                                            std::vector<int64_t>&      ldd,
                                            std::vector<int64_t>&      strideA,
                                            std::vector<int64_t>&      strideB,
                                            std::vector<int64_t>&      strideC,
                                            std::vector<int64_t>&      strideD,
                                            std::vector<GemmEpilogue>& epilogue,
                                            std::vector<GemmInputs>&   inputs,
                                            GemmProblemType&           problemtype)
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtGroupedGemmSetProblemFullV2Cpp");
        std::vector<rocblaslt::RocGemmEpilogueV2> rocepilogue;
        for(auto& e : epilogue)
        {
            rocepilogue.push_back(*reinterpret_cast<rocblaslt::RocGemmEpilogueV2*>(e.pimpl.get()));
        }

        std::vector<rocblaslt::RocGemmInputsV2> rocinputs;
        for(auto& i : inputs)
        {
            rocinputs.push_back(*reinterpret_cast<rocblaslt::RocGemmInputsV2*>(i.pimpl.get()));
        }
        GemmProblemType                              tmp = problemtype;
        std::vector<rocblaslt::RocGemmProblemTypeV2> rocproblemtype
            = {*reinterpret_cast<rocblaslt::RocGemmProblemTypeV2*>(tmp.pimpl.get())};
        auto status = RocBlasLtStatusToHIPStatus(
            rocblaslt_groupedgemm_create_cpp((rocblaslt_handle)m_handle,
                                             m,
                                             n,
                                             batch_count,
                                             k,
                                             lda,
                                             ldb,
                                             ldc,
                                             ldd,
                                             strideA,
                                             strideB,
                                             strideC,
                                             strideD,
                                             rocepilogue,
                                             rocinputs,
                                             rocproblemtype,
                                             m_data,
                                             m_gemm_count));
        if(status == HIPBLAS_STATUS_SUCCESS)
        {
            m_problem_types[0] = problemtype;
        }
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }

    hipblasStatus_t GroupedGemm::setProblem(std::vector<hipblasLtMatmulDesc_t>&   matmul_descr,
                                            std::vector<void*>&                   alpha,
                                            std::vector<void*>&                   A,
                                            std::vector<hipblasLtMatrixLayout_t>& matA,
                                            std::vector<void*>&                   B,
                                            std::vector<hipblasLtMatrixLayout_t>& matB,
                                            std::vector<void*>&                   beta,
                                            std::vector<void*>&                   C,
                                            std::vector<hipblasLtMatrixLayout_t>& matC,
                                            std::vector<void*>&                   D,
                                            std::vector<hipblasLtMatrixLayout_t>& matD)
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtGroupedGemmSetProblemCAPICpp");
        auto matmul_descr_groupedGemm
            = reinterpret_cast<std::vector<rocblaslt_matmul_desc>*>(&matmul_descr);
        auto matA_groupedGemm  = reinterpret_cast<std::vector<rocblaslt_matrix_layout>*>(&matA);
        auto matB_groupedGemm  = reinterpret_cast<std::vector<rocblaslt_matrix_layout>*>(&matB);
        auto matC_groupedGemm  = reinterpret_cast<std::vector<rocblaslt_matrix_layout>*>(&matC);
        auto matD_groupedGemm  = reinterpret_cast<std::vector<rocblaslt_matrix_layout>*>(&matD);
        auto A_groupedGemm     = reinterpret_cast<std::vector<const void*>*>(&A);
        auto B_groupedGemm     = reinterpret_cast<std::vector<const void*>*>(&B);
        auto C_groupedGemm     = reinterpret_cast<std::vector<const void*>*>(&C);
        auto alpha_groupedGemm = reinterpret_cast<std::vector<const void*>*>(&alpha);
        auto beta_groupedGemm  = reinterpret_cast<std::vector<const void*>*>(&beta);
        std::vector<rocblaslt::RocGemmProblemTypeV2> rocproblemtypes;
        auto                                         status = RocBlasLtStatusToHIPStatus(
            rocblaslt_groupedgemm_create_cpp((rocblaslt_handle)m_handle,
                                             *matmul_descr_groupedGemm,
                                             *alpha_groupedGemm,
                                             *A_groupedGemm,
                                             *matA_groupedGemm,
                                             *B_groupedGemm,
                                             *matB_groupedGemm,
                                             *beta_groupedGemm,
                                             *C_groupedGemm,
                                             *matC_groupedGemm,
                                             D,
                                             *matD_groupedGemm,
                                             rocproblemtypes,
                                             m_data,
                                             m_gemm_count));
        m_problem_types.clear();
        if(status == HIPBLAS_STATUS_SUCCESS)
        {
            m_problem_types.clear();
            for(auto& problemtype : rocproblemtypes)
            {
                m_problem_types.push_back(GemmProblemType(
                    *reinterpret_cast<GemmProblemType::GemmProblemTypeImpl*>(&problemtype)));
            }
        }
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }

    std::vector<GemmProblemType> GroupedGemm::getProblemTypes()
    {
        return m_problem_types;
    }

    HIPBLASLT_EXPORT hipblasStatus_t
        GroupedGemm::getDefaultValueForDeviceUserArguments(void* hostDeviceUserArgs)
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtGroupedGemmGetDefaultUserArgsCpp");
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto status   = RocBlasLtStatusToHIPStatus(rocblaslt_get_default_user_args(
            (rocblaslt_handle)m_handle, gemmType, m_data, hostDeviceUserArgs));
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }

    HIPBLASLT_EXPORT hipblasStatus_t GroupedGemm::run(void* deviceUserArgs, hipStream_t stream)
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtGroupedGemmRunCpp");
        if(m_gemm_count == 0)
        {
            rocblaslt::Debug::Instance().markerStop();
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto status   = RocBlasLtStatusToHIPStatus(rocblaslt_run_user_args_cpp(
            (rocblaslt_handle)m_handle, gemmType, m_data, deviceUserArgs, stream));
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }

    hipblasStatus_t matmulIsAlgoSupported(hipblasLtHandle_t       handle,
                                          hipblasLtMatmulDesc_t   matmulDesc,
                                          const void*             alpha,
                                          hipblasLtMatrixLayout_t Adesc,
                                          hipblasLtMatrixLayout_t Bdesc,
                                          const void*             beta,
                                          hipblasLtMatrixLayout_t Cdesc,
                                          hipblasLtMatrixLayout_t Ddesc,
                                          hipblasLtMatmulAlgo_t&  algo,
                                          size_t&                 workspaceSizeInBytes)
    try
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtMatMulIsAlgoSupportedCpp");
        auto status = RocBlasLtStatusToHIPStatus(
            rocblaslt_matmul_is_algo_supported((rocblaslt_handle)handle,
                                               (rocblaslt_matmul_desc)matmulDesc,
                                               alpha,
                                               (rocblaslt_matrix_layout)Adesc,
                                               (rocblaslt_matrix_layout)Bdesc,
                                               beta,
                                               (rocblaslt_matrix_layout)Cdesc,
                                               (rocblaslt_matrix_layout)Ddesc,
                                               (rocblaslt_matmul_algo*)&algo,
                                               &workspaceSizeInBytes));
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    std::string gemmType2String(GemmType type)
    {
        switch(type)
        {
        case GemmType::HIPBLASLT_GEMM:
            return "gemm";
        case GemmType::HIPBLASLT_GROUPED_GEMM:
            return "grouped gemm";
        }
    }

    hipblasStatus_t getAllAlgos(hipblasLtHandle_t                              handle,
                                GemmType                                       typeGemm,
                                hipblasOperation_t                             opA,
                                hipblasOperation_t                             opB,
                                hipDataType                                    typeA,
                                hipDataType                                    typeB,
                                hipDataType                                    typeC,
                                hipDataType                                    typeD,
                                hipblasComputeType_t                           typeCompute,
                                std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults)
    try
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtGetAllAlgosCpp");
        auto results
            = reinterpret_cast<std::vector<rocblaslt_matmul_heuristic_result>*>(&heuristicResults);
        results->clear();
        auto status = RocBlasLtStatusToHIPStatus(
            rocblaslt_matmul_get_all_algos_cpp((rocblaslt_handle)handle,
                                               static_cast<rocblaslt::RocGemmType>(typeGemm),
                                               opA,
                                               opB,
                                               typeA,
                                               typeB,
                                               typeC,
                                               typeD,
                                               (rocblaslt_compute_type)typeCompute,
                                               *results));
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    int getIndexFromAlgo(hipblasLtMatmulAlgo_t& algo)
    {
        int* algo_ptr = (int*)algo.data;

        return *algo_ptr;
    }

    std::string getSolutionNameFromAlgo(hipblasLtHandle_t handle, hipblasLtMatmulAlgo_t& algo)
    {
        int* algo_ptr = (int*)algo.data;
        if(*algo_ptr < 0)
        {
            return "";
        }
        auto rocalgo = reinterpret_cast<const rocblaslt_matmul_algo*>(&algo);
        return rocblaslt_get_solution_name_from_algo((rocblaslt_handle)handle, *rocalgo);
    }

    std::string getKernelNameFromAlgo(hipblasLtHandle_t handle, hipblasLtMatmulAlgo_t& algo)
    {
        int* algo_ptr = (int*)algo.data;
        if(*algo_ptr < 0)
        {
            return "";
        }
        auto rocalgo = reinterpret_cast<const rocblaslt_matmul_algo*>(&algo);
        return rocblaslt_get_kernel_name_from_algo((rocblaslt_handle)handle, *rocalgo);
    }

    hipblasStatus_t
        getAlgosFromIndex(hipblasLtHandle_t                              handle,
                          std::vector<int>&                              algoIndex,
                          std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults)
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtGetAlgosFromIndexCpp");
        auto results
            = reinterpret_cast<std::vector<rocblaslt_matmul_heuristic_result>*>(&heuristicResults);
        results->clear();
        auto status = RocBlasLtStatusToHIPStatus(rocblaslt_matmul_get_algos_from_index_cpp(
            (rocblaslt_handle)handle, algoIndex, *results));
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }

    hipblasStatus_t copyMatmul(hipblasLtMatmulDesc_t src, hipblasLtMatmulDesc_t dst)
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtCopyMatmulCpp");
        auto status = RocBlasLtStatusToHIPStatus(
            rocblaslt_copy_matmul((rocblaslt_matmul_desc)src, (rocblaslt_matmul_desc)dst));
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }

    int matmulIsTuned(hipblasLtHandle_t       handle,
                      hipblasLtMatmulDesc_t   matmulDesc,
                      hipblasLtMatrixLayout_t Adesc,
                      hipblasLtMatrixLayout_t Bdesc,
                      hipblasLtMatrixLayout_t Cdesc,
                      hipblasLtMatrixLayout_t Ddesc)
    {
        rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulIsTunedCpp");
        auto status = rocblaslt_matmul_is_tuned((rocblaslt_handle)handle,
                                                (rocblaslt_matmul_desc)matmulDesc,
                                                (rocblaslt_matrix_layout)Adesc,
                                                (rocblaslt_matrix_layout)Bdesc,
                                                (rocblaslt_matrix_layout)Cdesc,
                                                (rocblaslt_matrix_layout)Ddesc);
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }

} // End of namespace hipblasltext
