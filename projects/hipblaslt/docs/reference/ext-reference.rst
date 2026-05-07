.. meta::
   :description: hipBLASLtExt API reference
   :keywords: hipBLASLt, ROCm, library, API, reference

.. _ext-reference:

********************************
hipBLASLtExt API reference
********************************

hipBLASLt contains extension APIs with the namespace ``hipblaslt_ext``. They are only C++ compatible. The extensions support the following:

1. :ref:`GEMM <gemm>`

2. :ref:`Grouped GEMM <grouped-gemm>`

3. :ref:`Get all algorithms <get-all-algo>`

hipBLASLtExt datatypes reference
=================================

GemmType
-------------------------------------
.. doxygenenum:: GemmType

GemmProblemType
-------------------------------------
.. doxygenclass:: hipblaslt_ext::GemmProblemType
    :members:
    :protected-members:
    :private-members:

GemmEpilogue
-------------------------------------
.. doxygenclass:: hipblaslt_ext::GemmEpilogue
    :members:
    :protected-members:
    :private-members:

GemmInputs
-------------------------------------
.. doxygenclass:: hipblaslt_ext::GemmInputs
    :members:
    :protected-members:
    :private-members:

hipBLASLtExt GEMM class reference
=================================

GemmPreference
-------------------------------------
.. doxygenclass:: hipblaslt_ext::GemmPreference
    :members:
    :protected-members:
    :private-members:

GemmInstance
-------------------------------------
.. doxygenclass:: hipblaslt_ext::GemmInstance
    :members:
    :protected-members:
    :private-members:

Gemm
-------------------------------------
.. doxygenclass:: hipblaslt_ext::Gemm
    :members:
    :protected-members:
    :private-members:

GroupedGemm
-------------------------------------
.. doxygenclass:: hipblaslt_ext::GroupedGemm
    :members:
    :protected-members:
    :private-members:

hipBLASLtExt API reference
================================

getAllAlgos()
------------------------------------------
.. doxygenfunction:: getAllAlgos

getIndexFromAlgo()
------------------------------------------
.. doxygenfunction:: getIndexFromAlgo

getAlgosFromIndex()
------------------------------------------
.. doxygenfunction:: getAlgosFromIndex

matmulIsAlgoSupported()
------------------------------------------
.. doxygenfunction:: matmulIsAlgoSupported

hipblasLtExt usage
================================

Here are the three use cases supported by the hipBLASLtExt APIs.

.. _Gemm:

GEMM
--------------

hipblasLt has its own instance. You must assign the problem type when constructing or importing the problem from the hipBLAS API.

.. code-block:: c++

    HIPBLASLT_EXPORT explicit Gemm(hipblasLtHandle_t      handle,
                                   hipblasOperation_t     opA,
                                   hipblasOperation_t     opB,
                                   hipDataType      typeA,
                                   hipDataType      typeB,
                                   hipDataType      typeC,
                                   hipDataType      typeD,
                                   hipblasComputeType_t typeCompute);

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

After the instance is created, you can set the problem using the API.
The API might require the following structures:

   .. code-block:: c++

      struct GemmEpilogue
      {
         hipblasLtEpilogue_t mode = HIPBLASLT_EPILOGUE_DEFAULT;
         hipDataType   bias_data_type;
         int                 aux_ld;
         int                 aux_stride;
      };

*  ``setProblem`` APIs:

   .. code-block:: c++

      HIPBLASLT_EXPORT hipblasStatus_t setProblem(
         int64_t m, int64_t n, int64_t k, int64_t batch_count, GemmEpilogue& epilogue, GemmInputs& inputs);

You can set the leading dimensions and strides and reassign the data type using the following API:

.. code-block:: c++

    HIPBLASLT_EXPORT hipblasStatus_t setProblem(int64_t            m,
                                                int64_t            n,
                                                int64_t            k,
                                                int64_t            batch_count,
                                                int64_t            lda,
                                                int64_t            ldb,
                                                int64_t            ldc,
                                                int64_t            ldd,
                                                int64_t            strideA,
                                                int64_t            strideB,
                                                int64_t            strideC,
                                                int64_t            strideD,
                                                GemmEpilogue&    epilogue,
                                                GemmInputs&      inputs,
                                                GemmProblemType& problemtype);

You can import problems from the hipblasLt APIs after the instance is created.

.. note::

   This can overwrite the problem type of the instance.

.. code-block:: c++

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

You can retrieve heuristics and set kernel arguments with the instance. If the properties of the GEMM and the inputs don't change,
you can call the run API to launch the kernel directly.

.. code-block:: c++

    // Pseudo code
    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(1000000);
    // Default epilogue mode is HIPBLASLT_EPILOGUE_DEFAULT
    hipblaslt_ext::GemmEpilogue epilogue;
    hipblaslt_ext::GemmInputs inputs;
    inputs.setA(d_a);
    inputs.setB(d_b);
    inputs.setC(d_c);
    inputs.setD(d_d);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);

    hipblaslt_ext::Gemm gemm(handle,
                             HIPBLAS_OP_N,
                             HIPBLAS_OP_N,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIPBLAS_COMPUTE_32F);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
    gemm.setProblem(1, 1, 1, 1, epilogue, inputs); // m, n, k, batch
    gemm.algoGetHeuristic(gemm, pref, heuristic);
    gemm.initialize(heuristic[0].algo, d_workspace, stream);
    for(int i = 0; i < 10; i++)
    {
        gemm.run(stream);
    }

.. _grouped-gemm:

Grouped GEMM
--------------

``hipblasLtExt`` supports grouped GEMM. It shares the same class with normal GEMM.

After the problem is set, you can check the problem type using the function ``getGemmType()``.

.. code-block:: c++

    enum class GemmType
    {
        HIPBLASLT_GEMM             = 1,
        HIPBLASLT_GROUPED_GEMM     = 2
    };

The grouped GEMM class also includes the ``setProblem`` APIs.

.. code-block:: c++

    HIPBLASLT_EXPORT hipblasStatus_t setProblem(
        int64_t m, int64_t n, int64_t k, int64_t batch_count, GemmEpilogue& epilogue, GemmInputs& inputs);

    HIPBLASLT_EXPORT hipblasStatus_t setProblem(std::vector<int64_t>&        m,
                                                std::vector<int64_t>&        n,
                                                std::vector<int64_t>&        k,
                                                std::vector<int64_t>&        batch_count,
                                                std::vector<GemmEpilogue>& epilogue,
                                                std::vector<GemmInputs>&   inputs);

    HIPBLASLT_EXPORT hipblasStatus_t setProblem(std::vector<int64_t>&        m,
                                                std::vector<int64_t>&        n,
                                                std::vector<int64_t>&        k,
                                                std::vector<int64_t>&        batch_count,
                                                std::vector<int64_t>&        lda,
                                                std::vector<int64_t>&        ldb,
                                                std::vector<int64_t>&        ldc,
                                                std::vector<int64_t>&        ldd,
                                                std::vector<int64_t>&        strideA,
                                                std::vector<int64_t>&        strideB,
                                                std::vector<int64_t>&        strideC,
                                                std::vector<int64_t>&        strideD,
                                                std::vector<GemmEpilogue>& epilogue,
                                                std::vector<GemmInputs>&   inputs,
                                                GemmProblemType&           problemtype);

    HIPBLASLT_EXPORT hipblasStatus_t setProblem(std::vector<hipblasLtMatmulDesc_t>&   matmul_descr,
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

For the following API, the ``epilogue`` argument supports broadcasting to the length of the problem size
by duplicating the last element.

.. code-block:: c++

    HIPBLASLT_EXPORT hipblasStatus_t setProblem(std::vector<int64_t>&        m,
                                                std::vector<int64_t>&        n,
                                                std::vector<int64_t>&        k,
                                                std::vector<int64_t>&        batch_count,
                                                std::vector<int64_t>&        lda,
                                                std::vector<int64_t>&        ldb,
                                                std::vector<int64_t>&        ldc,
                                                std::vector<int64_t>&        ldd,
                                                std::vector<int64_t>&        strideA,
                                                std::vector<int64_t>&        strideB,
                                                std::vector<int64_t>&        strideC,
                                                std::vector<int64_t>&        strideD,
                                                std::vector<GemmEpilogue>& epilogue,
                                                std::vector<GemmInputs>&   inputs,
                                                GemmProblemType&           problemtype);

.. note::

   Only a ``problemtype`` size equal to 1 is currently supported. (This means only one ``GemmProblemType`` for all problems.)

.. code-block:: c++

    // Pseudo code
    std::vector<int64_t> m, n, k;
    // ...
    for(size_t i = 0; i < problem_size, i++)
    {
        // ...
    }
    std::vector<GemmProblemType> problemtypes;
    problemtypes.push_back(problemtype);
    groupedgemm.setProblem(m, n, k, batch_count, lda, ldb, ldc, ldd, strideA, strideB, strideC, strideD, epilogue, inputs, problemtypes);

The UserArguments structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Grouped GEMM supports the use of external device memory to run the kernel.
This is helpful if some of the arguments are from the output of the pervious kernel.
To change the size-related arguments ``m``, ``n``, ``k``, and ``batch``, see :ref:`Fixed MK <fixed-mk>`.

.. code-block:: c++

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
        void*    bias; //!< The bias input pointer.
        int      biasType; //!< The bias datatype. Only works if mode is set to bias related epilogues.
        uint32_t reserved;
        void*    e; //!< The aux input pointer. Only works if mode is set to aux related epilogues.
        uint32_t strideE1; //!< The aux leading dimension. Only works if mode is set to aux related epilogues.
        uint32_t strideE2; //!< The aux batch stride. Only works if mode is set to aux related epilogues.
        float    act0; //!< The activation value 1. Some activations might use it.
        float    act1; //!< The activation value 2.
        int      activationType; //!< The activation type.  Only works if mode is set to activation related epilogues.
    } __attribute__((packed));

hipBLASLt adds two functions to the ``UserArguments``-related API. The first API is a helper function that helps you initialize
the ``UserArguments`` structure from the saved problems inside the grouped GEMM object.
The second API is an overload function with an additional ``UserArguments`` device pointer input.

.. code-block:: c++

    HIPBLASLT_EXPORT hipblasStatus_t getDefaultValueForDeviceUserArguments(void* hostDeviceUserArgs);

    HIPBLASLT_EXPORT hipblasStatus_t run(void* deviceUserArgs, hipStream_t stream);

Here is a simple example that shows how this API works.

.. code-block:: c++

    // Pseudo code
    // Step 1: Get all algorithms
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(handle,
                                                     HIPBLASLT_GEMM,
                                                     HIPBLAS_OP_N,
                                                     HIPBLAS_OP_N,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     HIPBLAS_COMPUTE_32F,
                                                     heuristicResult));

    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(1000000);
    // Step 2: Setup problem
    std::vector<int64_t> m(gemm_count);
    std::vector<int64_t> n(gemm_count);
    std::vector<int64_t> k(gemm_count);
    std::vector<int64_t> batch_count(gemm_count);
    std::vector<hipblaslt_ext::GemmEpilogue> epilogue(gemm_count);
    std::vector<hipblaslt_ext::GemmInputs> inputs(gemm_count);
    for(int i = 0; i < gemm_count; i++)
    {
        m[i] = 1;
        n[i] = 1;
        k[i] = 1;
        batch_count[i] = 1;
        epilogue[i].setMode(HIPBLASLT_EPILOGUE_GELU);
        inputs[i].setA(d_a[i]);
        inputs[i].setB(d_b[i]);
        inputs[i].setC(d_c[i]);
        inputs[i].setD(d_d[i]);
        inputs[i].setAlpha(&alpha[i]);
        inputs[i].setBeta(&beta[i]);
    }

    // Step 3: Create grouped gemm instance
    hipblaslt_ext::GroupedGemm groupedGemm(handle,
                                           HIPBLAS_OP_N,
                                           HIPBLAS_OP_N,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIPBLAS_COMPUTE_32F);

    // Step 4: Set problem
    groupedGemm.setProblem(m, n, k, batch_count, epilogue, inputs); // m, n, k, batch

    // Step 5: Get default value from the instance
    hipblaslt_ext::UserArguments* dUAFloat = new hipblaslt_ext::UserArguments[gemm_count];
    groupedGemm.getDefaultValueForDeviceUserArguments((void*)dUAFloat);
    // Once you get the default value here, you can make several copies and change the values
    // from the host

    // Next copy them to the device memory
    hipblaslt_ext::UserArguments* d_dUAFloat = nullptr;
    hipMalloc(&d_dUAFloat, sizeof(hipblaslt_ext::UserArguments) * gemm_count);
    hipMemcpy(d_dUAFloat, dUAFloat, sizeof(hipblaslt_ext::UserArguments) * gemm_count, hipMemcpyHostToDevice);

    validIdx.clear();
    for(int j = 0; j < heuristicResult.size(); j++)
    {
        size_t workspace_size = 0;
        if(groupedGemm.isAlgoSupported(heuristicResult[j].algo, workspace_size)
           == HIPBLAS_STATUS_SUCCESS)
        {
            validIdx.push_back(j);
        }
    }

    // Step 6: Initialize and run
    if(validIdx.size() > 1)
    {
        groupedGemm.initialize(heuristicResult[validIdx[0]].algo, d_workspace, stream);
        for(int i = 0; i < 10; i++)
        {
            groupedGemm.run(userArgs, stream);
        }
    }

The base class (GemmInstance)
-----------------------------

This is the base class for ``Gemm`` and ``GroupedGemm``.

.. code-block:: c++

    // Gets heuristic from the instance.
    HIPBLASLT_EXPORT hipblasStatus_t algoGetHeuristic(const int                                      requestedAlgoCount,
                                                      const GemmPreference&                        pref,
                                                      std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults);

    // Returns SUCCESS if the algo is supported, also returns the required workspace size in bytes.
    HIPBLASLT_EXPORT hipblasStatus_t isAlgoSupported(hipblasLtMatmulAlgo_t& algo, size_t& workspaceSizeInBytes);

    // Initializes the instance before calling run. Requires every time the problem is set.
    HIPBLASLT_EXPORT hipblasStatus_t initialize(const hipblasLtMatmulAlgo_t& algo, void* workspace, bool useUserArgs = true, hipStream_t stream = 0);

    // Run the problem.
    HIPBLASLT_EXPORT hipblasStatus_t run(hipStream_t stream);

.. _get-all-algo:

Get all algorithms
------------------

Get all algorithms allows you to get all the algorithms for a specific problem type.
It requires the transpose of A, B, the data type of the inputs, and the compute type.

.. code-block:: c++

    HIPBLASLT_EXPORT
    hipblasStatus_t hipblaslt_ext::getAllAlgos(hipblasLtHandle_t                              handle,
                                               hipblasLtExtGemmTypeEnum_t                     typeGemm,
                                               hipblasOperation_t                             opA,
                                               hipblasOperation_t                             opB,
                                               hipDataType                              typeA,
                                               hipDataType                              typeB,
                                               hipDataType                              typeC,
                                               hipDataType                              typeD,
                                               hipblasComputeType_t                         typeCompute,
                                               std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults);

This API doesn't require a problem size or epilogue as input. It uses another API named ``isAlgoSupported`` to check
if the algorithm supports a problem.

.. code-block:: c++

    hipblaslt_ext::matmulIsAlgoSupported()
    gemm.isAlgoSupported()

The API returns the required workspace size in bytes upon successful completion.

.. code-block:: c++

    // Get all algorithms
    CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(handle,
                                                     HIPBLASLT_GEMM,
                                                     HIPBLAS_OP_N,
                                                     HIPBLAS_OP_N,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     HIPBLAS_COMPUTE_32F,
                                                     heuristicResult));

    validIdx.clear();
    for(int j = 0; j < heuristicResult.size(); j++)
    {
        size_t workspace_size = 0;
        if(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                matmul,
                                                &(alpha),
                                                matA,
                                                matB,
                                                &(beta),
                                                matC,
                                                matD,
                                                heuristicResult[j].algo,
                                                workspace_size)
           == HIPBLAS_STATUS_SUCCESS)
        {
            validIdx.push_back(j);
            heuristicResult[j].workspaceSize = workspace_size;
        }
        else
        {
            heuristicResult[j].workspaceSize = 0;
        }
    }

Algorithm index
-----------------

This extension API lets you to get the algorithm index using ``hipblasLtMatmulAlgo_t``.

.. code-block:: c++

    HIPBLASLT_EXPORT int getIndexFromAlgo(hipblasLtMatmulAlgo_t& algo);


You can use an index vector to retrieve the heuristic results.

.. code-block:: c++

    HIPBLASLT_EXPORT
    hipblasStatus_t
        getAlgosFromIndex(hipblasLtHandle_t                              handle,
                          std::vector<int>&                              algoIndex,
                          std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults);

Sample code
=================

This section contains some code samples that demonstrate the use cases of the extension APIs.

GEMM
---------

.. code-block:: c++

    // Pseudo code for gemm problem
    // Get all algorithms
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(handle,
                                                     HIPBLASLT_GEMM,
                                                     HIPBLAS_OP_N,
                                                     HIPBLAS_OP_N,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     HIPBLAS_COMPUTE_32F,
                                                     heuristicResult));

    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(1000000);
    hipblaslt_ext::GemmEpilogue epilogue;
    epilogue.setMode(HIPBLASLT_EPILOGUE_GELU);
    hipblaslt_ext::GemmInputs inputs;
    inputs.setA(d_a);
    inputs.setB(d_b);
    inputs.setC(d_c);
    inputs.setD(d_d);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);

    hipblaslt_ext::Gemm gemm(handle,
                             HIPBLAS_OP_N,
                             HIPBLAS_OP_N,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIPBLAS_COMPUTE_32F);

    gemm.setProblem(1, 1, 1, 1, epilogue, inputs); // m, n, k, batch

    validIdx.clear();
    for(int j = 0; j < heuristicResult.size(); j++)
    {
        size_t workspace_size = 0;
        if(gemm.isAlgoSupported(heuristicResult[j].algo, workspace_size)
           == HIPBLAS_STATUS_SUCCESS)
        {
            validIdx.push_back(j);
            heuristicResult[j].workspaceSize = workspace_size;
        }
        else
        {
            heuristicResult[j].workspaceSize = 0;
        }
    }

    if(validIdx.size() > 1)
    {
        gemm.initialize(heuristicResult[validIdx[0]].algo, d_workspace, stream);
        for(int i = 0; i < 10; i++)
        {
            gemm.run(stream);
        }
    }

Grouped GEMM
--------------

.. code-block:: c++

    // Pseudo code for grouped gemm problem
    // Get all algorithms
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(handle,
                                                     HIPBLASLT_GEMM,
                                                     HIPBLAS_OP_N,
                                                     HIPBLAS_OP_N,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     HIPBLAS_COMPUTE_32F,
                                                     heuristicResult));

    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(1000000);

    std::vector<int64_t> m(gemm_count);
    std::vector<int64_t> n(gemm_count);
    std::vector<int64_t> k(gemm_count);
    std::vector<int64_t> batch_count(gemm_count);
    std::vector<hipblaslt_ext::GemmEpilogue> epilogue(gemm_count);
    std::vector<hipblaslt_ext::GemmInputs> inputs(gemm_count);
    for(int i = 0; i < gemm_count; i++)
    {
        m[i] = 1;
        n[i] = 1;
        k[i] = 1;
        batch_count[i] = 1;
        epilogue[i].setMode(HIPBLASLT_EPILOGUE_GELU);
        inputs[i].setA(d_a[i]);
        inputs[i].setB(d_b[i]);
        inputs[i].setC(d_c[i]);
        inputs[i].setD(d_d[i]);
        inputs[i].setAlpha(&alpha[i]);
        inputs[i].setBeta(&beta[i]);
    }


    hipblaslt_ext::GroupedGemm groupedGemm(handle,
                                           HIPBLAS_OP_N,
                                           HIPBLAS_OP_N,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIPBLAS_COMPUTE_32F);

    groupedGemm.setProblem(m, n, k, batch_count, epilogue, inputs); // m, n, k, batch

    validIdx.clear();
    for(int j = 0; j < heuristicResult.size(); j++)
    {
        size_t workspace_size = 0;
        if(groupedGemm.isAlgoSupported(heuristicResult[j].algo, workspace_size)
           == HIPBLAS_STATUS_SUCCESS)
        {
            validIdx.push_back(j);
        }
    }

    if(validIdx.size() > 1)
    {
        groupedGemm.initialize(heuristicResult[validIdx[0]].algo, d_workspace, stream);
        for(int i = 0; i < 10; i++)
        {
            groupedGemm.run(stream);
        }
    }

Algorithm index
-----------------

.. code-block:: c++

    int index = hipblaslt_ext::getIndexFromAlgo(testResults[i].algo);
    // Save the index to disk or somewhere else for later use.

    // Get the index from previous state.
    std::vector<int> algoIndex{index};
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;
    // If the index is out of the bound of solutions, getAlgosFromIndex will return HIPBLAS_STATUS_INVALID_VALUE
    if(HIPBLAS_STATUS_INVALID_VALUE
        == hipblaslt_ext::getAlgosFromIndex(handle, algoIndex, heuristicResults))
    {
        std::cout << "Indexes are all out of bound." << std::endl;
        break;
    }

.. _fixed-mk:

[Grouped Gemm] Fixed MK
------------------------

The hipBLASLt extension supports changing the sizes (``m``, ``n``, ``k``, and ``batch``) from the device memory ``UserArguments``.
However, the setup is a bit different from the normal routing.

Sum of N
^^^^^^^^^

A sum of N needs to be used as an input for the grouped GEMM instance.

.. code-block:: c++

    {1000, 1, 1, 1}; // The array of N, the first element is the sum of N

    // Below is the values stored in "UserArguments"
    {256, 256, 1, 1}; // This is a valid configuration cause 256 + 256 + 1 + 1 < 1000
    {512, 512, 1, 1}; // This is NOT a valid configuration cause 512 + 512 + 1 + 1 > 1000

For example, consider a grouped GEMM with ``gemm_count = 4``. The sum of N must not exceed the "sum of N" set in the ``setProblem`` API.
In this mode, the first element is the "sum of N" in the array of Ns.

.. code-block:: c++

    // Pseudo code
    // Step 1: Get all algorithms
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(handle,
                                                     HIPBLASLT_GEMM,
                                                     HIPBLAS_OP_N,
                                                     HIPBLAS_OP_N,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     HIPBLAS_COMPUTE_32F,
                                                     heuristicResult));

    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(1000000);
    // Step 2: Setup problem
    std::vector<int64_t> m(gemm_count);
    std::vector<int64_t> n(gemm_count);
    std::vector<int64_t> k(gemm_count);
    std::vector<int64_t> batch_count(gemm_count);
    std::vector<hipblaslt_ext::GemmEpilogue> epilogue(gemm_count);
    std::vector<hipblaslt_ext::GemmInputs> inputs(gemm_count);

    // Step 2.1: Calculate sum of n
    int64_t sum_of_n = 0;
    for(int i = 0; i < gemm_count; i++)
    {
        sum_of_n += n_arr[i];
    }

    // {sum_of_n, 1, 1, 1, ...}; // The array of N, the first element is the sum of N
    for(int i = 0; i < gemm_count; i++)
    {
        m[i] = m_arr[i];
        if(i == 0)
            n[i] = sum_of_n;
        else
            n[i] = 1;
        k[i] = k_arr[i];
        batch_count[i] = 1;
        inputs[i].setA(d_a[i]);
        inputs[i].setB(d_b[i]);
        inputs[i].setC(d_c[i]);
        inputs[i].setD(d_d[i]);
        inputs[i].setAlpha(&alpha[i]);
        inputs[i].setBeta(&beta[i]);
    }

    // Step 3: Create grouped gemm instance
    hipblaslt_ext::GroupedGemm groupedGemm(handle,
                                           HIPBLAS_OP_N,
                                           HIPBLAS_OP_N,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIPBLAS_COMPUTE_32F);

    // Step 4: Set problem
    groupedGemm.setProblem(m, n, k, batch_count, epilogue, inputs); // m, n, k, batch

    // Step 5: Get default value from the instance
    hipblaslt_ext::UserArguments* dUAFloat = new hipblaslt_ext::UserArguments[gemm_count];
    groupedGemm.getDefaultValueForDeviceUserArguments((void*)dUAFloat);
    // Once you get the default value here, you can make several copies and change the values
    // from the host

    // Next Copy them to the device memory
    hipblaslt_ext::UserArguments* d_dUAFloat = nullptr;
    hipMalloc(&d_dUAFloat, sizeof(hipblaslt_ext::UserArguments) * gemm_count);
    hipMemcpy(d_dUAFloat, dUAFloat, sizeof(hipblaslt_ext::UserArguments) * gemm_count, hipMemcpyHostToDevice);

    validIdx.clear();
    for(int j = 0; j < heuristicResult.size(); j++)
    {
        size_t workspace_size = 0;
        if(groupedGemm.isAlgoSupported(heuristicResult[j].algo, workspace_size)
           == HIPBLAS_STATUS_SUCCESS)
        {
            validIdx.push_back(j);
        }
    }

    int threads = 256;
    int blocks  = ceil((double)gemm_count / threads);

    // Step 6: Initialize and run
    if(validIdx.size() > 1)
    {
        groupedGemm.initialize(heuristicResult[validIdx[0]].algo, d_workspace);
        for(int i = 0; i < 10; i++)
        {
            hipLaunchKernelGGL(kernelUpdateN,
                                dim3(blocks),
                                dim3(threads),
                                0,
                                stream,
                                gemm_count,
                                d_dUAFloat,
                                d_n_vec);  // d_n_vec is a device pointer with Ns
            groupedGemm.run(userArgs, stream);
        }
    }

    // .....

    __global__ void kernelUpdateN(uint32_t gemm_count, void* userArgs, int32_t* sizes_n)
    {
    uint64_t id = hipBlockIdx_x * 256 + hipThreadIdx_x;

    if(id >= gemm_count)
        return;

    hipblaslt_ext::UserArguments* dUAFloat = static_cast<hipblaslt_ext::UserArguments*>(userArgs);
    dUAFloat[id].n                         = sizes_n[id];
    }
