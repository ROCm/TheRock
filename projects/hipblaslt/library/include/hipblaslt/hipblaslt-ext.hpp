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

/*! \file
 *  \brief hipblaslt-ext.hpp provides general matrix-matrix operations with
 *  C++ style flexible API to let user set attributes for solution selection.
 */

#pragma once
#include "hipblaslt/hipblaslt.h"

#include <memory>
#include <vector>

namespace hipblaslt_ext
{
    class GemmInstance;
    class Gemm;
    class GroupedGemm;

    using HipBufferDeleter = hipError_t (*)(void*);
    using HipBufferPtr     = std::unique_ptr<void, HipBufferDeleter>;

    /*! \ingroup types_module
     *  \brief It is an enumerated type used to specific the type of the gemm problem in hipblasLtExt APIs.
     */
    enum class GemmType
    {
        HIPBLASLT_GEMM         = 1,
        HIPBLASLT_GROUPED_GEMM = 2,
    };

    /*! \ingroup types_module
     *  \brief hipblasLt extension preference for gemm problems.
     *
     * \details Currently only supports setting max workspace size.
     */
    class GemmPreference
    {
    public:
        HIPBLASLT_EXPORT GemmPreference();
        HIPBLASLT_EXPORT ~GemmPreference();

        HIPBLASLT_EXPORT                 GemmPreference(const GemmPreference& pref);
        HIPBLASLT_EXPORT GemmPreference& operator=(const GemmPreference& pref);

        HIPBLASLT_EXPORT                 GemmPreference(GemmPreference&& pref);
        HIPBLASLT_EXPORT GemmPreference& operator=(GemmPreference&& pref);

        /*! \ingroup library_module
         *  \brief This function sets the max workspace size.
         *
         *  @param[in]
         *  workspaceBytes  Set the max workspace size in bytes.
         */
        HIPBLASLT_EXPORT void setMaxWorkspaceBytes(size_t workspaceBytes);

        /*! \ingroup library_module
         *  \brief This function returns the set max workspace size.
         *
         *  \retval size_t Returns the set max workspace size.
         */
        HIPBLASLT_EXPORT const size_t getMaxWorkspaceBytes() const;

    private:
        friend GemmInstance;
        class GemmPreferenceImpl;
        std::unique_ptr<GemmPreferenceImpl> pimpl;
    };

    [[deprecated("GemmPreferenceV2 is deprecated, use GemmPreference instead.")]]
    typedef GemmPreference GemmPreferenceV2;

    /*! \ingroup types_module
     *  \brief hipblasLt extension ProblemType for gemm problems.
     *
     * \details This structure sets the problem type of a gemm problem.
     */
    class GemmProblemType
    {
    private:
        friend Gemm;
        friend GroupedGemm;
        class GemmProblemTypeImpl;
        std::unique_ptr<GemmProblemTypeImpl> pimpl;

    public:
        HIPBLASLT_EXPORT GemmProblemType();
        HIPBLASLT_EXPORT GemmProblemType(hipblasOperation_t   opA,
                                         hipblasOperation_t   opB,
                                         hipDataType          typeA,
                                         hipDataType          typeB,
                                         hipDataType          typeC,
                                         hipDataType          typeD,
                                         hipblasComputeType_t typeCompute);
        HIPBLASLT_EXPORT ~GemmProblemType();

        HIPBLASLT_EXPORT                  GemmProblemType(const GemmProblemTypeImpl& impl);
        HIPBLASLT_EXPORT GemmProblemType& operator=(const GemmProblemTypeImpl& impl);
        HIPBLASLT_EXPORT                  GemmProblemType(const GemmProblemType& type);
        HIPBLASLT_EXPORT GemmProblemType& operator=(const GemmProblemType& type);

        HIPBLASLT_EXPORT                  GemmProblemType(GemmProblemType&& type);
        HIPBLASLT_EXPORT GemmProblemType& operator=(GemmProblemType&& type);

        HIPBLASLT_EXPORT void setOpA(hipblasOperation_t op); //!< Set the A martix transpose.
        HIPBLASLT_EXPORT void setOpB(hipblasOperation_t op); //!< Set the B matrix transpose.
        HIPBLASLT_EXPORT void setTypeA(hipDataType type); //!< Set the A matrix datatype.
        HIPBLASLT_EXPORT void setTypeB(hipDataType type); //!< Set the B matrix datatype.
        HIPBLASLT_EXPORT void setTypeC(hipDataType type); //!< Set the C matrix datatype.
        HIPBLASLT_EXPORT void setTypeD(hipDataType type); //!< Set the D matrix datatype.
        HIPBLASLT_EXPORT void
            setTypeCompute(hipblasComputeType_t type); //!< Set the compute datatype.
        HIPBLASLT_EXPORT void setOrderA(hipblasLtOrder_t order); //!< Set the A martix data order.
        HIPBLASLT_EXPORT void setOrderB(hipblasLtOrder_t order); //!< Set the B matrix data order.

        HIPBLASLT_EXPORT hipblasOperation_t   getOpA() const; //!< The A matrix transpose.
        HIPBLASLT_EXPORT hipblasOperation_t   getOpB() const; //!< The B matrix transpose.
        HIPBLASLT_EXPORT hipDataType          getTypeA() const; //!< The A matrix datatype.
        HIPBLASLT_EXPORT hipDataType          getTypeB() const; //!< The B matrix datatype.
        HIPBLASLT_EXPORT hipDataType          getTypeC() const; //!< The C matrix datatype.
        HIPBLASLT_EXPORT hipDataType          getTypeD() const; //!< The D matrix datatype.
        HIPBLASLT_EXPORT hipblasComputeType_t getTypeCompute() const; //!< The compute datatype.
        HIPBLASLT_EXPORT hipblasLtOrder_t     getOrderA() const; //!< The A matrix data order.
        HIPBLASLT_EXPORT hipblasLtOrder_t     getOrderB() const; //!< The B matrix data order.
    };

    [[deprecated("GemmProblemTypeV2 is deprecated, use GemmProblemType instead.")]]
    typedef GemmProblemType GemmProblemTypeV2;

    /*! \ingroup types_module
     *  \brief hipblasLt extension Epilogue for gemm problems.
     *
     * \details This class sets the epilogue of a gemm problem.
     */
    class GemmEpilogue
    {
    public:
        HIPBLASLT_EXPORT GemmEpilogue();
        HIPBLASLT_EXPORT ~GemmEpilogue();

        HIPBLASLT_EXPORT               GemmEpilogue(const GemmEpilogue& epilogue);
        HIPBLASLT_EXPORT GemmEpilogue& operator=(const GemmEpilogue& epilogue);

        HIPBLASLT_EXPORT               GemmEpilogue(GemmEpilogue&& epilogue);
        HIPBLASLT_EXPORT GemmEpilogue& operator=(GemmEpilogue&& epilogue);

        HIPBLASLT_EXPORT void
            setMode(hipblasLtEpilogue_t mode); //!< Set the mode of epilogue. Default is gemm.
        HIPBLASLT_EXPORT void setBiasDataType(
            hipDataType
                biasDataType); //!< Set the bias datatype. Only works if mode is set to bias related epilogues.
        HIPBLASLT_EXPORT void setAuxDataType(
            hipDataType
                auxDataType); //!< Set the aux datatype. Only works if mode is set to aux related epilogues.
        HIPBLASLT_EXPORT void setAuxLeadingDimension(
            int auxLeadingDimension); //!< Set the aux leading dimension. Only works if mode is set to aux related epilogues.
        HIPBLASLT_EXPORT void setAuxBatchStride(
            int auxBatchStride); //!< Set the aux batch stride. Only works if mode is set to aux related epilogues.
        HIPBLASLT_EXPORT void
            setScalingAType(hipblasLtMatmulMatrixScale_t
                                scalingAType); //!< Only works if DataTypeA = DataTypeB = FP8.
        HIPBLASLT_EXPORT void
            setScalingBType(hipblasLtMatmulMatrixScale_t
                                scalingBType); //!< Only works if DataTypeA = DataTypeB = FP8.
        HIPBLASLT_EXPORT void setAct0(float act0); //!< Set first extra argument for activation function.
        HIPBLASLT_EXPORT void setAct1(float act1); //!< Set second extra argument for activation function.

        HIPBLASLT_EXPORT hipblasLtEpilogue_t
                                     getMode() const; //!< The mode of epilogue. Default is gemm.
        HIPBLASLT_EXPORT hipDataType getBiasDataType()
            const; //!< The bias datatype. Only works if mode is set to bias related epilogues.
        HIPBLASLT_EXPORT hipDataType getAuxDataType()
            const; //!< The aux datatype. Only works if mode is set to aux related epilogues.
        HIPBLASLT_EXPORT int getAuxLeadingDimension()
            const; //!< The aux leading dimension. Only works if mode is set to aux related epilogues.
        HIPBLASLT_EXPORT int getAuxBatchStride()
            const; //!< The aux batch stride. Only works if mode is set to aux related epilogues.
        HIPBLASLT_EXPORT hipblasLtMatmulMatrixScale_t getScalingAType()
            const; //!< 0 is scalar, 1 is vector. Only works if DataTypeA = DataTypeB = FP8.
        HIPBLASLT_EXPORT hipblasLtMatmulMatrixScale_t getScalingBType()
            const; //!< 0 is scalar, 1 is vector. Only works if DataTypeA = DataTypeB = FP8.
        HIPBLASLT_EXPORT float getAct0(); //!< first extra argument for activation function.
        HIPBLASLT_EXPORT float getAct1(); //!< second extra argument for activation function.
    private:
        friend Gemm;
        friend GroupedGemm;
        class GemmEpilogueImpl;
        std::unique_ptr<GemmEpilogueImpl> pimpl;
    };

    [[deprecated("GemmEpilogueV2 is deprecated, use GemmEpilogue instead.")]]
    typedef GemmEpilogue GemmEpilogueV2;

    struct GemmTuning
    {
    public:
        HIPBLASLT_EXPORT GemmTuning();
        HIPBLASLT_EXPORT ~GemmTuning();

        HIPBLASLT_EXPORT             GemmTuning(const GemmTuning& tuning);
        HIPBLASLT_EXPORT GemmTuning& operator=(const GemmTuning& tuning);

        HIPBLASLT_EXPORT             GemmTuning(GemmTuning&& tuning);
        HIPBLASLT_EXPORT GemmTuning& operator=(GemmTuning&& tuning);

        HIPBLASLT_EXPORT void setSplitK(
            uint16_t
                splitK); //!< Set the value of splitK, 0 is off (use the splitK inside the solution).
        HIPBLASLT_EXPORT void setWgm(
            int16_t
                wgm); //!< Set the value of workgroup mapping, 0 is off (use the workgroup mapping inside the solution).

        HIPBLASLT_EXPORT uint16_t getSplitK() const; //!< Value of splitK.
        HIPBLASLT_EXPORT int16_t  getWgm() const; //!< Value of workgroup mapping.
    private:
        friend GemmInstance;
        class GemmTuningImpl;
        std::unique_ptr<GemmTuningImpl> pimpl;
    };

    [[deprecated("GemmTuningV2 is deprecated, use GemmTuning instead.")]]
    typedef GemmTuning GemmTuningV2;

    /*! \ingroup types_module
     *  \brief hipblasLt extension Inputs for gemm problems.
     *
     * \details This class sets the input pointers of a gemm problem.
     */
    class GemmInputs
    {
    public:
        HIPBLASLT_EXPORT GemmInputs();
        HIPBLASLT_EXPORT ~GemmInputs();

        HIPBLASLT_EXPORT             GemmInputs(const GemmInputs& input);
        HIPBLASLT_EXPORT GemmInputs& operator=(const GemmInputs& input);

        HIPBLASLT_EXPORT             GemmInputs(GemmInputs&& input);
        HIPBLASLT_EXPORT GemmInputs& operator=(GemmInputs&& input);

        HIPBLASLT_EXPORT void setA(const void* a); //!< Set the a matrix input pointer.
        HIPBLASLT_EXPORT void setB(const void* b); //!< Set the b matrix input pointer.
        HIPBLASLT_EXPORT void setC(const void* c); //!< Set the c matrix input pointer.
        HIPBLASLT_EXPORT void setD(const void* d); //!< Set the d matrix input pointer.
        HIPBLASLT_EXPORT void setAlpha(const void* alpha); //!< Set the alpha value.
        HIPBLASLT_EXPORT void setBeta(const void* beta); //!< Set the beta value.
        HIPBLASLT_EXPORT void setBias(const void* bias); //!< Set the bias input pointer.
        HIPBLASLT_EXPORT void setScaleA(const void* scaleA); //!< Set the Scale A input pointer.
        HIPBLASLT_EXPORT void setScaleB(const void* scaleB); //!< Set the Scale B input pointer.
        HIPBLASLT_EXPORT void setScaleC(const void* scaleC); //!< Set the Scale C input pointer.
        HIPBLASLT_EXPORT void setScaleD(const void* scaleD); //!< Set the Scale D input pointer.
        HIPBLASLT_EXPORT void
            setScaleAux(const void* scaleAux); //!< Set the Scale AUX input pointer.
        HIPBLASLT_EXPORT void setScaleAlphaVec(
            const void* scaleAlphaVec); //!< Set the scaleAlpha vector input pointer.
        HIPBLASLT_EXPORT void setAux(const void* aux); //!< Set the aux input pointer.
        HIPBLASLT_EXPORT void setAmaxD(const void* amaxD); //!< Set the AmaxD input pointer.

        HIPBLASLT_EXPORT const void* getA() const; //!< The a matrix input pointer.
        HIPBLASLT_EXPORT const void* getB() const; //!< The b matrix input pointer.
        HIPBLASLT_EXPORT const void* getC() const; //!< The c matrix input pointer.
        HIPBLASLT_EXPORT const void* getD() const; //!< The d matrix input pointer.
        HIPBLASLT_EXPORT const void* getAlpha() const; //!< The alpha value.
        HIPBLASLT_EXPORT const void* getBeta() const; //!< The beta value.
        HIPBLASLT_EXPORT const void* getBias() const; //!< The bias input pointer.
        HIPBLASLT_EXPORT const void* getScaleA() const; //!< The Scale A input pointer.
        HIPBLASLT_EXPORT const void* getScaleB() const; //!< The Scale B input pointer.
        HIPBLASLT_EXPORT const void* getScaleC() const; //!< The Scale C input pointer.
        HIPBLASLT_EXPORT const void* getScaleD() const; //!< The Scale D input pointer.
        HIPBLASLT_EXPORT const void* getScaleAux() const; //!< The Scale AUX input pointer.
        HIPBLASLT_EXPORT const void*
            getScaleAlphaVec() const; //!< The scaleAlpha vector input pointer.
        HIPBLASLT_EXPORT const void* getAux() const; //!< The aux input pointer
        HIPBLASLT_EXPORT const void* getAmaxD() const; //!< The AmaxD input pointer

    private:
        friend Gemm;
        friend GroupedGemm;
        class GemmInputsImpl;
        std::unique_ptr<GemmInputsImpl> pimpl;
    };

    [[deprecated("GemmInputsV2 is deprecated, use GemmInputs instead.")]]
    typedef GemmInputs GemmInputsV2;

    /*! \ingroup types_module
     *  \brief hipblasLt extension GPU inputs for gemm problems.
     *
     * \details This structure sets the input gpu pointers of a gemm problem.
     * Only supports solutions loading arguments from global memory.
     */

    struct UserArguments
    {
        uint32_t m; //!< size m
        uint32_t n; //!< size n
        uint32_t batch; //!< size batch
        uint32_t k; //!< size k
        void*    d; //!< The d matrix input pointer.
        void*    c; //!< The c matrix input pointer.
        void*    a; //!< The a matrix input pointer.
        void*    b; //!< The b matrix input pointer.
        uint32_t strideD1; //!< The d leading dimension.
        uint32_t strideD2; //!< The d batch stride
        uint32_t strideC1; //!< The c leading dimension.
        uint32_t strideC2; //!< The c batch stride
        uint32_t strideA1; //!< The a leading dimension.
        uint32_t strideA2; //!< The a batch stride
        uint32_t strideB1; //!< The b leading dimension.
        uint32_t strideB2; //!< The b batch stride
        int8_t   alpha[16]; //!< The alpha value.
        int8_t   beta[16]; //!< The beta value.
        // Epilogue inputs
        void* scaleA; //!< The scaleA input pointer.
        void* scaleB; //!< The scaleB input pointer.
        void* scaleC; //!< The scaleC input pointer.
        void* scaleD; //!< The scaleD input pointer.
        void* scaleAlphaVec; //!< The scaleAlpha vector input pointer.
        void* bias; //!< The bias input pointer.
        int   biasType; //!< The bias datatype. Only works if mode is set to bias related epilogues.
        uint32_t reserved;
        void*    e; //!< The aux input pointer. Only works if mode is set to aux related epilogues.
        uint32_t
            strideE1; //!< The aux leading dimension. Only works if mode is set to aux related epilogues.
        uint32_t
            strideE2; //!< The aux batch stride. Only works if mode is set to aux related epilogues.
        float act0; //!< The activation value 1. Some activations might use it.
        float act1; //!< The activation value 2.
        int   activationType; //!< The activation type.  Only works if mode is set to activation related epilogues.
    } __attribute__((packed));

    /*! \ingroup types_module
     *  \brief hipblasLt extension instance for gemm problems.
     */
    class GemmInstance
    {
    public:
        HIPBLASLT_EXPORT virtual ~GemmInstance(){};
        GemmInstance(const GemmInstance& rhs)                             = delete;
        GemmInstance&                  operator=(const GemmInstance& rhs) = delete;
        HIPBLASLT_EXPORT               GemmInstance(GemmInstance&& rhs) noexcept;
        HIPBLASLT_EXPORT GemmInstance& operator=(GemmInstance&& rhs) noexcept;

        /*! \ingroup library_module
        *  \brief Retrieve the possible algorithms
        *
        *  \details
        *  This function retrieves the possible algorithms for the matrix multiply
        * operation hipblasLtMatmul() function with the given data and compute tpye.
        * The output is placed in heuristicResult in the order of increasing
        * estimated compute time.
        *
        *  @param[in]
        *  requestedAlgoCount  number of requested algorithms.
        *  @param[in]
        *  pref hipblasLt extension preference for gemm problems.
        *  @param[out]
        *  heuristicResults    The algorithm heuristic vector.
        *
        *  \retval HIPBLAS_STATUS_SUCCESS           If query was successful. Inspect
        * heuristicResults.size > 0, but may heuristicResults.size < requestedAlgoCount
        * state for the status of the results. \retval HIPBLAS_STATUS_NOT_SUPPORTED
        * If no heuristic function available for current configuration.
        * \retval HIPBLAS_STATUS_INVALID_VALUE If no solution is found.
        */
        HIPBLASLT_EXPORT
        hipblasStatus_t
            algoGetHeuristic(const int                                      requestedAlgoCount,
                             const GemmPreference&                          pref,
                             std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults);

        /*! \ingroup library_module
        *  \brief Check if the algorithm supports the problem. (For hipblaslt extension API)
        *
        *  \details
        *  This function updates the problem saved inside the algorithm if the problem is
        * supported. The required workspaceSizeInBytes is also returned.
        *
        *  @param[in]
        *  algo The algorithm heuristic.
        *  @param[out]
        *  workspaceSizeInBytes Return the required workspace size.
        *
        *  \retval HIPBLAS_STATUS_SUCCESS           If query was successful. The problem is
        * supported by the algorithm.
        * results. \retval HIPBLAS_STATUS_INVALID_VALUE     The problem is not supported.
        */
        HIPBLASLT_EXPORT
        hipblasStatus_t isAlgoSupported(hipblasLtMatmulAlgo_t& algo, size_t& workspaceSizeInBytes);

        /*! \ingroup library_module
        *  \brief Check if the algorithm supports the problem. (For hipblaslt extension API)
        *
        *  \details
        *  This function updates the problem saved inside the algorithm if the problem is
        * supported. The required workspaceSizeInBytes is also returned.
        *
        *  @param[in]
        *  algo The algorithm heuristic.
        *  @param[in]
        *  tuning The tuning parameters.
        *  @param[out]
        *  workspaceSizeInBytes Return the required workspace size.
        *
        *  \retval HIPBLAS_STATUS_SUCCESS           If query was successful. The problem is
        * supported by the algorithm.
        * results. \retval HIPBLAS_STATUS_INVALID_VALUE     The problem is not supported.
        */
        HIPBLASLT_EXPORT
        hipblasStatus_t isAlgoSupported(hipblasLtMatmulAlgo_t& algo,
                                        GemmTuning&            tuning,
                                        size_t&                workspaceSizeInBytes);

        /*! \ingroup library_module
         *  \brief This function sets the max workspace size.
         *
         *  @param[in]
         *  workspaceBytes  Set the max workspace size in bytes.
         */
        HIPBLASLT_EXPORT void setMaxWorkspaceBytes(size_t workspaceBytes);

        /*! \ingroup library_module
         *  \brief This function returns the set max workspace size.
         *
         *  \retval size_t Returns the set max workspace size.
         */
        HIPBLASLT_EXPORT const size_t getMaxWorkspaceBytes() const;

        /*! \ingroup library_module
        *  \brief Create kernel arguments from a given hipblaslt_ext::GemmInstance.
        *
        *  \details
        *  This function creates kernel arguments from a given hipblaslt_ext::GemmInstance
        *  then saves the arguments inside the instance.
        *
        *  @param[in]
        *  algo                    Handle for matrix multiplication algorithm to be
        * used. See hipblaslt.h::hipblasLtMatmulAlgo_t . When NULL, an implicit heuristics query
        * with default search preferences will be performed to determine actual
        * algorithm to use.
        *  @param[in]
        *  workspace               Pointer to the workspace buffer allocated in the GPU
        * memory. Pointer must be 16B aligned (that is, lowest 4 bits of address must
        * be 0).
        *  @param[in]
        *  useUserArgs                Use user args, this does not affect vanilla gemm.
        * (May be deprecated in the future)
        *  @param[in]
        *  stream                  The HIP stream where all the GPU work will be
        * submitted. (May be deprecated in the future)
        *
        *  \retval HIPBLAS_STATUS_SUCCESS           If the operation completed
        * successfully. \retval HIPBLAS_STATUS_INVALID_VALUE If the gemm_count = 0 or
        * workspace is null but workspaceBytes is greater than zero.
        * Note that workspaceBytes should be set with setMaxWorkspaceBytes.
        */
        HIPBLASLT_EXPORT
        hipblasStatus_t initialize(const hipblasLtMatmulAlgo_t& algo,
                                   void*                        workspace,
                                   bool                         useUserArgs = true,
                                   hipStream_t                  stream      = 0);

        /*! \ingroup library_module
        *  \brief Create kernel arguments from a given hipblaslt_ext::GemmInstance.
        *
        *  \details
        *  This function creates kernel arguments from a given hipblaslt_ext::GemmInstance
        *  then saves the arguments inside the instance.
        *
        *  @param[in]
        *  algo                    Handle for matrix multiplication algorithm to be
        * used. See hipblaslt.h::hipblasLtMatmulAlgo_t . When NULL, an implicit heuristics query
        * with default search preferences will be performed to determine actual
        * algorithm to use.
        *  @param[in]
        *  tuning                  Structure with user tuning parameters. Note that not every algo
        * supports user tuning parameters. Will return HIPBLAS_STATUS_INVALID_VALUE if not supported.
        * be 0).
        *  @param[in]
        *  workspace               Pointer to the workspace buffer allocated in the GPU
        * memory. Pointer must be 16B aligned (that is, lowest 4 bits of address must
        * be 0).
        *  @param[in]
        *  useUserArgs                Use user args, this does not affect vanilla gemm.
        * (May be deprecated in the future)
        *  @param[in]
        *  stream                  The HIP stream where all the GPU work will be
        * submitted. (May be deprecated in the future)
        *
        *  \retval HIPBLAS_STATUS_SUCCESS           If the operation completed
        * successfully. \retval HIPBLAS_STATUS_INVALID_VALUE If the gemm_count = 0 or
        * workspace is null but workspaceBytes is greater than zero.
        * Note that workspaceBytes should be set with setMaxWorkspaceBytes.
        */
        HIPBLASLT_EXPORT
        hipblasStatus_t initialize(const hipblasLtMatmulAlgo_t& algo,
                                   GemmTuning&                  tuning,
                                   void*                        workspace,
                                   bool                         useUserArgs = true,
                                   hipStream_t                  stream      = 0);

        /*! \ingroup library_module
        *  \brief Execute the kernel arguments stored inside the hipblaslt_ext::GemmInstance.
        *
        *  @param[in]
        *  stream                  The HIP stream where all the GPU work will be
        *  @param[in]
        *  start                   The HIP event which will record the start of the kernel
        *  @param[in]
        *  stop                    The HIP event which will record the end of the kernel
        * submitted.
        *
        *  \retval HIPBLAS_STATUS_SUCCESS           If the operation completed
        * successfully.
        */
        HIPBLASLT_EXPORT
        hipblasStatus_t
            run(hipStream_t stream, hipEvent_t start = nullptr, hipEvent_t stop = nullptr);

        HIPBLASLT_EXPORT GemmType getGemmType();
        HIPBLASLT_EXPORT size_t   getGemmCount();

        HIPBLASLT_EXPORT std::string getSolutionName();
        HIPBLASLT_EXPORT std::string getKernelName();

    protected:
        /*! \ingroup library_module
        *  \brief Constructor of GemmInstance.
        */
        HIPBLASLT_EXPORT explicit GemmInstance(hipblasLtHandle_t handle, GemmType type);

        GemmType m_gemm_type;
        size_t   m_gemm_count = 0;

        std::vector<GemmProblemType> m_problem_types;

        hipblasLtHandle_t     m_handle;
        std::shared_ptr<void> m_data;

        size_t m_workspace_bytes = 0;
    };

    /*! \ingroup types_module
     *  \brief hipblasLt extension instance for gemm.
     *
     * \details The instance can be used to create arguments to compute the matrix
     * multiplication of matrices A and B to produce the output matrix D, according
     * to the following operation: \p D = \p alpha*( \p A *\p B) + \p beta*( \p C ),
     * where \p A, \p B, and \p C are input matrices, and \p alpha and \p beta are
     * input scalars.
     */
    class Gemm : public GemmInstance
    {
    public:
        /*! \ingroup library_module
        *  \brief Constructor
        *
        *  \details
        *  This function set the problem from hipblasLt structures. For more information
        * about the structures, see hipblasLtMatmul for more information.
        *
        *  @param[in]
        *  handle                     The handle from hipBLASLt.
        *  @param[in]
        *  opA,opB                    The transpose type of matrix A, B
        *  @param[in]
        *  typeA,typeB,typeC,typeD    The data type of matrix A, B, C, D
        *  @param[in]
        *  typeCompute                The compute type of the gemm problem
        */
        HIPBLASLT_EXPORT explicit Gemm(hipblasLtHandle_t    handle,
                                       hipblasOperation_t   opA,
                                       hipblasOperation_t   opB,
                                       hipDataType          typeA,
                                       hipDataType          typeB,
                                       hipDataType          typeC,
                                       hipDataType          typeD,
                                       hipblasComputeType_t typeCompute);

        /*! \ingroup library_module
        *  \brief Constructor that sets the gemm problem from hipblasLt structures
        *
        *  \details
        *  This constructor sets the problem from hipblasLt structures. For more information
        * about the structures, see hipblasLtMatmul for more information.
        *
        *  @param[in]
        *  handle                  The handle from hipBLASLt.
        *  @param[in]
        *  matmul_descr              Handle to a previously created matrix multiplication
        * descriptor of type \ref hipblasLtMatmulDesc_t .
        *  @param[in]
        *  alpha,beta              Pointers to the scalars used in the multiplication.
        *  @param[in]
        *  matA,matB,matC,matD Handles to the previously created matrix layout
        * descriptors of the type \ref hipblasLtMatrixLayout_t .
        *  @param[in]
        *  A,B,C                   Pointers to the GPU memory associated with the
        * corresponding descriptors \p matA, \p matB and \p matC .
        *  @param[out]
        *  D                       Pointer to the GPU memory associated with the
        * descriptor \p matD .
        */
        HIPBLASLT_EXPORT explicit Gemm(hipblasLtHandle_t       handle,
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
                                       hipblasLtMatrixLayout_t matD);

        Gemm(const Gemm&) = delete;
        HIPBLASLT_EXPORT       Gemm(Gemm&&) noexcept;
        Gemm&                  operator=(const Gemm&) = delete;
        HIPBLASLT_EXPORT Gemm& operator=(Gemm&&) noexcept;

        /*! \ingroup library_module
        *  \brief Sets the problem for a gemm problem.
        *
        *  \details
        *  This function sets the problem with m, n, k, batch_count. It uses the problem type sets
        *  from the constructor.
        *
        *  @param[in]
        *  m,n,k                      The problem size.
        *  @param[in]
        *  batch_count                The batch count.
        *  @param[in]
        *  epilogue                   The class that controls the epilogue.
        *  @param[in]
        *  inputs                     The inputs of the problem.
        *
        *  \retval HIPBLAS_STATUS_SUCCESS           If the operation completed
        * successfully. \retval HIPBLAS_STATUS_EXECUTION_FAILED  If HIP reported an
        * execution error from the device. \retval HIPBLAS_STATUS_ARCH_MISMATCH     If
        * the configured operation cannot be run using the selected device. \retval
        * HIPBLAS_STATUS_NOT_SUPPORTED     If the current implementation on the
        * selected device doesn't support the configured operation. \retval
        * HIPBLAS_STATUS_INVALID_VALUE     If the parameters are unexpectedly NULL, in
        * conflict or in an impossible configuration.
        *  \retval HIBLAS_STATUS_NOT_INITIALIZED    If hipBLASLt handle has not been
        * initialized.
        */
        HIPBLASLT_EXPORT hipblasStatus_t setProblem(int64_t       m,
                                                    int64_t       n,
                                                    int64_t       k,
                                                    int64_t       batch_count,
                                                    GemmEpilogue& epilogue,
                                                    GemmInputs&   inputs);

        /*! \ingroup library_module
        *  \brief Sets the problem for a gemm problem.
        *
        *  \details
        *  This function sets the problem with m, n, k, batch_count. It uses the problem type sets
        *  from the constructor.
        *
        *  @param[in]
        *  m,n,k                            The problem size.
        *  @param[in]
        *  batch_count                      The batch count.
        *  @param[in]
        *  lda,ldb,ldc,ldd                  The leading dimensions of the matrix.
        *  @param[in]
        *  strideA,strideB,strideC,strideD  The batch stride of the matrix.
        *  @param[in]
        *  epilogue                         The structure that controls the epilogue.
        *  @param[in]
        *  inputs                           The inputs of the problem.
        *  @param[in]
        *  problemtype                      The structure that sets the problem type of a gemm problem.
        *
        *  \retval HIPBLAS_STATUS_SUCCESS           If the operation completed
        * successfully. \retval HIPBLAS_STATUS_EXECUTION_FAILED  If HIP reported an
        * execution error from the device. \retval HIPBLAS_STATUS_ARCH_MISMATCH     If
        * the configured operation cannot be run using the selected device. \retval
        * HIPBLAS_STATUS_NOT_SUPPORTED     If the current implementation on the
        * selected device doesn't support the configured operation. \retval
        * HIPBLAS_STATUS_INVALID_VALUE     If the parameters are unexpectedly NULL, in
        * conflict or in an impossible configuration.
        *  \retval HIBLAS_STATUS_NOT_INITIALIZED    If hipBLASLt handle has not been
        * initialized.
        */
        HIPBLASLT_EXPORT hipblasStatus_t setProblem(int64_t          m,
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
                                                    GemmProblemType& problemtype);

        /*! \ingroup library_module
        *  \brief Sets the gemm problem from hipblasLt structures
        *
        *  \details
        *  This function sets the problem from hipblasLt structures. For more information
        * about the structures, see hipblasLtMatmul for more information.
        *
        *  @param[in]
        *  matmul_descr              Handle to a previously created matrix multiplication
        * descriptor of type \ref hipblasLtMatmulDesc_t .
        *  @param[in]
        *  alpha,beta              Pointers to the scalars used in the multiplication.
        *  @param[in]
        *  matA,matB,matC,matD Handles to the previously created matrix layout
        * descriptors of the type \ref hipblasLtMatrixLayout_t .
        *  @param[in]
        *  A,B,C                   Pointers to the GPU memory associated with the
        * corresponding descriptors \p matA, \p matB and \p matC .
        *  @param[out]
        *  D                       Pointer to the GPU memory associated with the
        * descriptor \p matD .
        *
        *  \retval HIPBLAS_STATUS_SUCCESS           If the operation completed
        * successfully. \retval HIPBLAS_STATUS_EXECUTION_FAILED  If HIP reported an
        * execution error from the device. \retval HIPBLAS_STATUS_ARCH_MISMATCH     If
        * the configured operation cannot be run using the selected device. \retval
        * HIPBLAS_STATUS_NOT_SUPPORTED     If the current implementation on the
        * selected device doesn't support the configured operation. \retval
        * HIPBLAS_STATUS_INVALID_VALUE     If the parameters are unexpectedly NULL, in
        * conflict or in an impossible configuration.
        *  \retval HIBLAS_STATUS_NOT_INITIALIZED    If hipBLASLt handle has not been
        * initialized.
        */
        HIPBLASLT_EXPORT hipblasStatus_t setProblem(hipblasLtMatmulDesc_t   matmul_descr,
                                                    const void*             alpha,
                                                    const void*             A,
                                                    hipblasLtMatrixLayout_t matA,
                                                    const void*             B,
                                                    hipblasLtMatrixLayout_t matB,
                                                    const void*             beta,
                                                    const void*             C,
                                                    hipblasLtMatrixLayout_t matC,
                                                    void*                   D,
                                                    hipblasLtMatrixLayout_t matD);

        HIPBLASLT_EXPORT GemmProblemType getProblemTypes();
    };

    /*! \ingroup types_module
     *  \brief hipblasLt extension instance for grouped gemm.
     *
     * \details The instance can be used to create arguments to compute the matrix
     * multiplication of matrices A and B to produce the output matrix D, according
     * to the following operation: \p D = \p alpha*( \p A *\p B) + \p beta*( \p C ),
     * where \p A, \p B, and \p C are input matrices, and \p alpha and \p beta are
     * input scalars.
     */
    class GroupedGemm : public GemmInstance
    {
    public:
        /*! \ingroup library_module
        *  \brief Constructor
        *
        *  \details
        *  This function set the problem from hipblasLt structures. For more information
        * about the structures, see hipblasLtMatmul for more information.
        *
        *  @param[in]
        *  handle                     The handle from hipBLASLt.
        *  @param[in]
        *  opA,opB                    The transpose type of matrix A, B
        *  @param[in]
        *  typeA,typeB,typeC,typeD    The data type of matrix A, B, C, D
        *  @param[in]
        *  typeCompute                The compute type of the gemm problem
        */
        HIPBLASLT_EXPORT explicit GroupedGemm(hipblasLtHandle_t    handle,
                                              hipblasOperation_t   opA,
                                              hipblasOperation_t   opB,
                                              hipDataType          typeA,
                                              hipDataType          typeB,
                                              hipDataType          typeC,
                                              hipDataType          typeD,
                                              hipblasComputeType_t typeCompute);
        GroupedGemm(const GroupedGemm&) = delete;
        HIPBLASLT_EXPORT              GroupedGemm(GroupedGemm&&) noexcept;
        GroupedGemm&                  operator=(const GroupedGemm&) = delete;
        HIPBLASLT_EXPORT GroupedGemm& operator=(GroupedGemm&&) noexcept;

        /*! \ingroup library_module
        *  \brief Constructor that sets the grouped gemm problem from hipblasLt structures
        *
        *  \details
        *  This constructor sets the problem from hipblasLt structures. For more information
        * about the structures, see hipblasLtMatmul for more information.
        *
        *  @param[in]
        *  handle                  The handle from hipBLASLt.
        *  @param[in]
        *  matmul_descr              Vectors of handle to a previously created matrix
        * multiplication descriptor of type \ref hipblasLtMatmulDesc_t .
        *  @param[in]
        *  alpha,beta              Vectors of float used in the multiplication.
        *  @param[in]
        *  matA,matB,matC,matD Vectors of handle to the previously created matrix
        * layout descriptors of the type \ref hipblasLtMatrixLayout_t .
        *  @param[in]
        *  A,B,C                   Vectors of pointer to the GPU memory associated
        * with the corresponding descriptors \p matA, \p matB and \p matC .
        *  @param[out]
        *  D                       Vector of pointer to the GPU memory associated with
        * the descriptor \p matD .
        */
        HIPBLASLT_EXPORT explicit GroupedGemm(hipblasLtHandle_t                     handle,
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
                                              std::vector<hipblasLtMatrixLayout_t>& matD);

        /*! \ingroup library_module
        *  \brief Sets the problem for a gemm problem.
        *
        *  \details
        *  This function sets the problem with m, n, k, batch_count. It uses the problem type sets
        *  from the constructor.
        *
        *  @param[in]
        *  m,n,k                      The problem size in vector.
        *  @param[in]
        *  batch_count                The batch count in vector.
        *  @param[in]
        *  epilogue                   The structure in vector that controls the epilogue.
        *  @param[in]
        *  inputs                     The inputs in vector of the problem.
        *
        *  \retval HIPBLAS_STATUS_SUCCESS           If the operation completed
        * successfully. \retval HIPBLAS_STATUS_EXECUTION_FAILED  If HIP reported an
        * execution error from the device. \retval HIPBLAS_STATUS_ARCH_MISMATCH     If
        * the configured operation cannot be run using the selected device. \retval
        * HIPBLAS_STATUS_NOT_SUPPORTED     If the current implementation on the
        * selected device doesn't support the configured operation. \retval
        * HIPBLAS_STATUS_INVALID_VALUE     If the parameters are unexpectedly NULL, in
        * conflict or in an impossible configuration.
        *  \retval HIBLAS_STATUS_NOT_INITIALIZED    If hipBLASLt handle has not been
        * initialized.
        */
        HIPBLASLT_EXPORT hipblasStatus_t setProblem(std::vector<int64_t>&      m,
                                                    std::vector<int64_t>&      n,
                                                    std::vector<int64_t>&      k,
                                                    std::vector<int64_t>&      batch_count,
                                                    std::vector<GemmEpilogue>& epilogue,
                                                    std::vector<GemmInputs>&   inputs);

        /*! \ingroup library_module
        *  \brief Sets the problem for a gemm problem.
        *
        *  \details
        *  This function sets the problem with m, n, k, batch_count. It uses the problem type sets
        *  from the constructor.
        *
        *  @param[in]
        *  m,n,k                            The problem size in vector.
        *  @param[in]
        *  batch_count                      The batch count in vector.
        *  @param[in]
        *  lda,ldb,ldc,ldd                  The leading dimensions in vector of the matrix.
        *  @param[in]
        *  strideA,strideB,strideC,strideD  The batch stride in vector of the matrix.
        *  @param[in]
        *  epilogue                         The structure in vector that controls the epilogue.
        *  @param[in]
        *  inputs                           The inputs in vector of the problem.
        *  @param[in]
        *  problemtype                      The structure that sets the problem type
        * of a gemm problem.
        *
        *  \retval HIPBLAS_STATUS_SUCCESS           If the operation completed
        * successfully. \retval HIPBLAS_STATUS_EXECUTION_FAILED  If HIP reported an
        * execution error from the device. \retval HIPBLAS_STATUS_ARCH_MISMATCH     If
        * the configured operation cannot be run using the selected device. \retval
        * HIPBLAS_STATUS_NOT_SUPPORTED     If the current implementation on the
        * selected device doesn't support the configured operation. \retval
        * HIPBLAS_STATUS_INVALID_VALUE     If the parameters are unexpectedly NULL, in
        * conflict or in an impossible configuration.
        *  \retval HIBLAS_STATUS_NOT_INITIALIZED    If hipBLASLt handle has not been
        * initialized.
        */
        HIPBLASLT_EXPORT hipblasStatus_t setProblem(std::vector<int64_t>&      m,
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
                                                    GemmProblemType&           problemtype);

        /*! \ingroup library_module
        *  \brief Sets the grouped gemm problem from hipblasLt structures
        *
        *  \details
        *  This function sets the problem from hipblasLt structures. For more information
        * about the structures, see hipblasLtMatmul for more information.
        *
        *  @param[in]
        *  matmul_descr              Vectors of handle to a previously created matrix
        * multiplication descriptor of type \ref hipblasLtMatmulDesc_t .
        *  @param[in]
        *  alpha,beta              Vectors of float used in the multiplication.
        *  @param[in]
        *  matA,matB,matC,matD Vectors of handle to the previously created matrix
        * layout descriptors of the type \ref hipblasLtMatrixLayout_t .
        *  @param[in]
        *  A,B,C                   Vectors of pointer to the GPU memory associated
        * with the corresponding descriptors \p matA, \p matB and \p matC .
        *  @param[out]
        *  D                       Vector of pointer to the GPU memory associated with
        * the descriptor \p matD .
        *
        *  \retval HIPBLAS_STATUS_SUCCESS           If the operation completed
        * successfully. \retval HIPBLAS_STATUS_EXECUTION_FAILED  If HIP reported an
        * execution error from the device. \retval HIPBLAS_STATUS_ARCH_MISMATCH     If
        * the configured operation cannot be run using the selected device. \retval
        * HIPBLAS_STATUS_NOT_SUPPORTED     If the current implementation on the
        * selected device doesn't support the configured operation. \retval
        * HIPBLAS_STATUS_INVALID_VALUE     If the parameters are unexpectedly NULL, in
        * conflict or in an impossible configuration.
        *  \retval HIBLAS_STATUS_NOT_INITIALIZED    If hipBLASLt handle has not been
        * initialized.
        */
        HIPBLASLT_EXPORT hipblasStatus_t
            setProblem(std::vector<hipblasLtMatmulDesc_t>&   matmul_descr,
                       std::vector<void*>&                   alpha,
                       std::vector<void*>&                   A,
                       std::vector<hipblasLtMatrixLayout_t>& matA,
                       std::vector<void*>&                   B,
                       std::vector<hipblasLtMatrixLayout_t>& matB,
                       std::vector<void*>&                   beta,
                       std::vector<void*>&                   C,
                       std::vector<hipblasLtMatrixLayout_t>& matC,
                       std::vector<void*>&                   D,
                       std::vector<hipblasLtMatrixLayout_t>& matD);

        HIPBLASLT_EXPORT std::vector<GemmProblemType> getProblemTypes();

        /*! \ingroup library_module
        *  \brief A helper function to initialize DeviceUserArguments using the set problem(s)
        * saved in the gemm object.
        *
        *  @param[in]
        *  hostDeviceUserArgs The DeviceUserArguments struture allocated in host. Note that
        * the user must put the correct type of the DeviceUserArguments.
        *
        *  \retval HIPBLAS_STATUS_SUCCESS           If the operation completed successfully.
        */
        HIPBLASLT_EXPORT hipblasStatus_t
            getDefaultValueForDeviceUserArguments(void* hostDeviceUserArgs);

        using GemmInstance::run;

        /*! \ingroup library_module
        *  \brief Run the kernel using DeviceUserArguments
        *
        *  @param[in]
        *  deviceUserArgs          Pointer to the DeviceUserArguments buffer allocated
        * in the GPU memory. Pointer must be 16B aligned (that is, lowest 4 bits of
        *  @param[in]
        *  stream                  The HIP stream where all the GPU work will be
        * submitted.
        *
        *  \retval HIPBLAS_STATUS_SUCCESS           If the operation completed
        * successfully. \retval HIPBLAS_STATUS_INVALID_VALUE If the gemm_count = 0.
        */
        HIPBLASLT_EXPORT hipblasStatus_t run(void* deviceUserArgs, hipStream_t stream);
    };

    /*******************************************************************************
     * Ext APIs
     ******************************************************************************/

    HIPBLASLT_EXPORT std::string gemmType2String(GemmType type);

    /*! \ingroup library_module
     *  \brief Retrieve the possible algorithms
     *
     *  \details
     *  This function retrieves the possible algorithms for the matrix multiply
     * operation hipblasLtMatmul() function with the given data and compute tpye.
     * The output is placed in heuristicResults in the order of increasing
     * estimated compute time. It should use matmulIsAlgoSupported() to check if
     * the algorithm support the problem before execute hipblasLtMatmul().
     *
     *  @param[in]
     *  handle                  Pointer to the allocated hipBLASLt handle for the
     * hipBLASLt context. See \ref hipblasLtHandle_t .
     *  @param[in]
     *  typeGemm Gemm type. ex. GEMM, GROUPED_GEMM.
     *  @param[in]
     *  opA, opB Transpose settings of A, B.
     *  @param[in]
     *  typeA,typeB,typeC,typeD The data type of matrix A, B, C, D.
     *  @param[in]
     *  typeCompute             The compute type.
     *  @param[out]
     *  heuristicResults The algorithm heuristic vector.
     *
     *  \retval HIPBLAS_STATUS_SUCCESS           If query was successful. Inspect
     * returnedAlgoCount > 0.state for the status of the
     * results. \retval HIPBLAS_STATUS_NOT_SUPPORTED     If no heuristic function
     * available for current configuration. \retval HIPBLAS_STATUS_INVALID_VALUE If
     * no solution is found.
     */
    HIPBLASLT_EXPORT
    hipblasStatus_t getAllAlgos(hipblasLtHandle_t                              handle,
                                GemmType                                       typeGemm,
                                hipblasOperation_t                             opA,
                                hipblasOperation_t                             opB,
                                hipDataType                                    typeA,
                                hipDataType                                    typeB,
                                hipDataType                                    typeC,
                                hipDataType                                    typeD,
                                hipblasComputeType_t                           typeCompute,
                                std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults);

    /*! \ingroup library_module
     *  \brief Retrieve the algorithm index
     *
     *  @param[in]
     *  algo    The algorithm.
     *
     *  \retval int The index of the algorithm, can be used to get heuristic
     * results from \ref getAlgosFromIndex. Note that the index may not be valid
     * if the algo struct is not initialized properly.
     */
    HIPBLASLT_EXPORT int getIndexFromAlgo(hipblasLtMatmulAlgo_t& algo);

    /*! \ingroup library_module
     *  \brief Retrieve the solution name
     *
     *  @param[in]
     *  handle  Pointer to the allocated hipBLASLt handle for the
     * hipBLASLt context. See \ref hipblasLtHandle_t .
     *  @param[in]
     *  algo    The algorithm.
     *
     *  \retval std::string The solution name of the algorithm, can be used to
     * get heuristic results from \ref getAlgosFromIndex. Returns "" if the
     * index stored in algo < 0. Note that the string may not be valid if the
     * algo struct is not initialized properly.
     */
    HIPBLASLT_EXPORT std::string getSolutionNameFromAlgo(hipblasLtHandle_t      handle,
                                                         hipblasLtMatmulAlgo_t& algo);

    /*! \ingroup library_module
     *  \brief Retrieve the kernel name
     *
     *  @param[in]
     *  handle  Pointer to the allocated hipBLASLt handle for the
     * hipBLASLt context. See \ref hipblasLtHandle_t .
     *  @param[in]
     *  algo    The algorithm.
     *
     *  \retval std::string The kernel name of the algorithm, can be used to
     * get heuristic results from \ref getAlgosFromIndex. Returns "" if the
     * index stored in algo < 0. Note that the string may not be valid if the
     * algo struct is not initialized properly.
     */
    HIPBLASLT_EXPORT std::string getKernelNameFromAlgo(hipblasLtHandle_t      handle,
                                                       hipblasLtMatmulAlgo_t& algo);

    /*! \ingroup library_module
     *  \brief Retrieve the possible algorithms
     *
     *  \details
     *  This function retrieves the possible algorithms for the matrix multiply
     * operation hipblasLtMatmul() function with the given index.
     * The output is placed in heuristicResult in the order of increasing
     * estimated compute time. A recorded solution index cannot be used across
     * different verison of library. It should use matmulIsAlgoSupported() to
     * check if the algorithm support the problem before execute hipblasLtMatmul().
     *
     *  @param[in]
     *  handle                  Pointer to the allocated hipBLASLt handle for the
     * hipBLASLt context. See \ref hipblasLtHandle_t .
     *  @param[in]
     *  algoIndex               The algorithm index vector.
     *  @param[out]
     *  heuristicResults         The algorithm heuristic vector.
     *
     *  \retval HIPBLAS_STATUS_SUCCESS           If query was successful. Inspect
     * heuristicResults.size() > 0.state for the status of the
     * results. \retval HIPBLAS_STATUS_NOT_SUPPORTED     If no heuristic function
     * available for current configuration. \retval HIPBLAS_STATUS_INVALID_VALUE If
     * query indexes are all out of bound of solution map.
     */
    HIPBLASLT_EXPORT
    hipblasStatus_t
        getAlgosFromIndex(hipblasLtHandle_t                              handle,
                          std::vector<int>&                              algoIndex,
                          std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults);

    /*! \ingroup library_module
     *  \brief Check if the algorithm supports the problem. (For hipblasLt API)
     *
     *  \details
     *  This function updates the problem saved inside the algorithm if the problem is
     * supported. The required workspaceSizeInBytes is also returned.
     *
     *  @param[in]
     *  handle                  Pointer to the allocated hipBLASLt handle for the
     * hipBLASLt context. See \ref hipblasLtHandle_t .
     *  @param[in]
     *  matmulDesc              Handle to a previously created matrix multiplication
     * descriptor of type \ref hipblasLtMatmulDesc_t .
     *  @param[in]
     *  alpha,beta              Pointers to the scalars used in the multiplication.
     *  @param[in]
     *  Adesc,Bdesc,Cdesc,Ddesc Handles to the previously created matrix layout
     * descriptors of the type \ref hipblasLtMatrixLayout_t .
     *  @param[in]
     *  algo The algorithm heuristic.
     *  @param[out]
     *  workspaceSizeInBytes Return the required workspace size.
     *
     *  \retval HIPBLAS_STATUS_SUCCESS           If query was successful. The problem is
     * supported by the algorithm.
     * results. \retval HIPBLAS_STATUS_INVALID_VALUE     The problem is not supported.
     */
    HIPBLASLT_EXPORT
    hipblasStatus_t matmulIsAlgoSupported(hipblasLtHandle_t       handle,
                                          hipblasLtMatmulDesc_t   matmulDesc,
                                          const void*             alpha,
                                          hipblasLtMatrixLayout_t Adesc,
                                          hipblasLtMatrixLayout_t Bdesc,
                                          const void*             beta,
                                          hipblasLtMatrixLayout_t Cdesc,
                                          hipblasLtMatrixLayout_t Ddesc,
                                          hipblasLtMatmulAlgo_t&  algo,
                                          size_t&                 workspaceSizeInBytes);

    /*! \ingroup library_module
     *  \brief Copy the settings from A matmul to B matmul.
     *
     *  @param[in]
     *  src Source matmul.
     *  @param[out]
     *  dst Return the copied matmul content.
     *
     *  \retval HIPBLAS_STATUS_SUCCESS           If query was successful. The problem is
     * supported by the algorithm.
     * results. \retval HIPBLAS_STATUS_NOT_INITIALIZED Source or dest matmul not initialized.
     */
    HIPBLASLT_EXPORT
    hipblasStatus_t copyMatmul(hipblasLtMatmulDesc_t src, hipblasLtMatmulDesc_t dst);

    HIPBLASLT_EXPORT
    int matmulIsTuned(hipblasLtHandle_t       handle,
                      hipblasLtMatmulDesc_t   matmulDesc,
                      hipblasLtMatrixLayout_t Adesc,
                      hipblasLtMatrixLayout_t Bdesc,
                      hipblasLtMatrixLayout_t Cdesc,
                      hipblasLtMatrixLayout_t Ddesc);
} // End of namespace hipblasltext
