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
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt_init.hpp>
#include <iostream>
#include <numeric>
#include <string>

#include <llvm/ObjectYAML/YAML.h>

#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"

extern "C" __global__ void flush_icache()
{
    asm __volatile__("s_icache_inv \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t" ::
                         :);
}

class Layer
{
public:
    enum TYPE
    {
        GEMM,
        FLUSH,
        UNKNOWN
    } type;
    std::string name;

    // User input
    int   m;
    int   n;
    int   k;
    int   batch;
    float alpha;
    float beta;

    hipblaslt_ext::GemmProblemType problem;
    hipblaslt_ext::GemmEpilogue    epilogue;

    // Internal switch
    bool is_using_bias = false;

    // Internal data
    int64_t            ws_size;
    void*              ws   = NULL;
    void*              a    = NULL;
    void*              b    = NULL;
    void*              c    = NULL;
    void*              d    = NULL;
    void*              bias = NULL;
    std::vector<void*> d_a, d_b, d_c, d_d, d_bias;

    // Internal hipblaslt_ext::Gem instance
    std::shared_ptr<std::vector<hipblaslt_ext::Gemm>> gemms;

    // Internal related
    int algo_index = -1;
    int block_count;

    void setData(int                         m,
                 int                         n,
                 int                         k,
                 int                         batch,
                 float                       alpha,
                 float                       beta,
                 hipblasOperation_t          op_a,
                 hipblasOperation_t          op_b,
                 hipDataType                 type_a,
                 hipDataType                 type_b,
                 hipDataType                 type_c,
                 hipDataType                 type_d,
                 hipblasComputeType_t        type_compute,
                 hipblaslt_ext::GemmEpilogue epilogue)
    {
        this->m        = m;
        this->n        = n;
        this->k        = k;
        this->batch    = batch;
        this->alpha    = alpha;
        this->beta     = beta;
        this->epilogue = epilogue;

        this->problem.setOpA(op_a);
        this->problem.setOpB(op_b);
        this->problem.setTypeA(type_a);
        this->problem.setTypeB(type_b);
        this->problem.setTypeC(type_c);
        this->problem.setTypeD(type_d);
        this->problem.setTypeCompute(type_compute);
    }

    void initBlock(int block)
    {
        block_count = block;
        for(int b = 0; b < block_count; b++)
        {
            d_a.push_back(NULL);
            d_b.push_back(NULL);
            d_c.push_back(NULL);
            d_d.push_back(NULL);
            d_bias.push_back(NULL);
        }
    }
    ~Layer()
    {
        if(a)
            CHECK_HIP_ERROR(hipFree(a));
        if(b)
            CHECK_HIP_ERROR(hipFree(b));
        if(c)
            CHECK_HIP_ERROR(hipFree(c));
        if(d)
            CHECK_HIP_ERROR(hipFree(d));
        if(bias)
            CHECK_HIP_ERROR(hipFree(bias));
        if(ws)
            CHECK_HIP_ERROR(hipFree(ws));
        for(int b = 0; b < block_count; b++)
        {
            if(d_a[b])
                CHECK_HIP_ERROR(hipFree(d_a[b]));
            if(d_b[b])
                CHECK_HIP_ERROR(hipFree(d_b[b]));
            if(d_c[b])
                CHECK_HIP_ERROR(hipFree(d_c[b]));
            if(d_d[b])
                CHECK_HIP_ERROR(hipFree(d_d[b]));
            if(d_bias[b])
                CHECK_HIP_ERROR(hipFree(d_bias[b]));
        }
    }
};

Layer::TYPE string2LayerType(std::string& value)
{
    return value == "GEMM"    ? Layer::TYPE::GEMM
           : value == "FLUSH" ? Layer::TYPE::FLUSH
                              : Layer::TYPE::UNKNOWN;
}

int32_t type2Size(hipDataType type)
{
    switch(type)
    {
    case hipDataType::HIP_R_8F_E4M3_FNUZ:
    case hipDataType::HIP_R_8F_E5M2_FNUZ:
    case hipDataType::HIP_R_8F_E4M3:
    case hipDataType::HIP_R_8F_E5M2:
        return sizeof(float) / 4;
    case hipDataType::HIP_R_32F:
        return sizeof(float);
    case hipDataType::HIP_R_16F:
        return sizeof(float) / 2;
    default:
        return 0;
    }
    return 0;
}

void initData(hipDataType type, void* data, int m, int n, int lda, int stride, int batch_count)
{
    switch(type)
    {
    case hipDataType::HIP_R_8F_E4M3_FNUZ:
    {
        hipblaslt_init_cos<hipblaslt_f8_fnuz>(
            (hipblaslt_f8_fnuz*)data, m, n, lda, stride, batch_count);
    }
    break;
    case hipDataType::HIP_R_8F_E4M3:
    {
        hipblaslt_init_cos<hipblaslt_f8>((hipblaslt_f8*)data, m, n, lda, stride, batch_count);
    }
    break;
    case hipDataType::HIP_R_16F:
    {
        hipblaslt_init_cos<hipblasLtHalf>((hipblasLtHalf*)data, m, n, lda, stride, batch_count);
    }
    break;
    default:
        exit(1);
    }
    return;
}

void initAndCopy(
    void** data, std::vector<void*>& d_data, int m, int n, int b, hipDataType type, int block_count)
{
    int64_t size = m * n * b * type2Size(type);
    CHECK_HIP_ERROR(hipHostMalloc(data, size));
    initData(type, *data, m, n, m, m * n, b);
    for(int b = 0; b < block_count; b++)
    {
        CHECK_HIP_ERROR(hipMalloc(&d_data[b], size));
        CHECK_HIP_ERROR(hipMemcpy(d_data[b], *data, size, hipMemcpyHostToDevice));
    }
}

void initNoCopy(
    void** data, std::vector<void*>& d_data, int m, int n, int b, hipDataType type, int block_count)
{
    int64_t size = m * n * b * type2Size(type);
    CHECK_HIP_ERROR(hipHostMalloc(data, size));
    for(int b = 0; b < block_count; b++)
        CHECK_HIP_ERROR(hipMalloc(&d_data[b], size));
}

class LayerConfigIOGeneralSettings
{
public:
    bool     print_kernel_info  = false;
    uint32_t rotating           = 0; // Size in MB
    uint32_t cold_iters         = 1000;
    uint32_t iters              = 10;
    int64_t  max_workspace_size = 128 * 1024 * 1024;
    bool     graph_mode         = false;
};

class LayerConfigIO
{
public:
    LayerConfigIOGeneralSettings gs;
    std::vector<Layer>           layer;
};

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(Layer)

namespace llvm
{
    namespace yaml
    {
        template <>
        struct MappingTraits<LayerConfigIOGeneralSettings>
        {
            static void mapping(IO& io, LayerConfigIOGeneralSettings& lc)
            {
                io.mapOptional("PrintKernelInfo", lc.print_kernel_info);
                io.mapOptional("Rotating", lc.rotating);
                io.mapOptional("ColdIter", lc.cold_iters);
                io.mapOptional("Iter", lc.iters);
                io.mapOptional("MaxWorkspaceSize", lc.max_workspace_size);
                io.mapOptional("UseGraphMode", lc.graph_mode);
            }
        };
        template <>
        struct MappingTraits<Layer>
        {
            static void mapping(IO& io, Layer& l)
            {
                std::string type;
                io.mapRequired("LayerType", type);
                l.type = string2LayerType(type);
                if(l.type == Layer::TYPE::UNKNOWN)
                {
                    std::cout << "Unknown Gemm type (GEMM/FLUSH)." << std::endl;
                    exit(1);
                }
                else if(l.type == Layer::TYPE::FLUSH)
                {
                    return;
                }
                io.mapOptional("Name", l.name);

                // Basic information
                std::vector<uint32_t> sizes;
                io.mapRequired("Size", sizes);
                if(sizes.size() != 3 && sizes.size() != 4)
                {
                    std::cout << "Size must be [m,n,k,b] or [m,n,k]" << std::endl;
                    exit(1);
                }
                l.m     = sizes[0];
                l.n     = sizes[1];
                l.k     = sizes[2];
                l.batch = sizes.size() == 4 ? sizes[3] : 1;

                io.mapRequired("Alpha", l.alpha);
                io.mapRequired("Beta", l.beta);

                // Problem type
                bool isTranspose = false;
                io.mapRequired("TransposeA", isTranspose);
                l.problem.setOpA(isTranspose ? HIPBLAS_OP_T : HIPBLAS_OP_N);
                io.mapRequired("TransposeB", isTranspose);
                l.problem.setOpB(isTranspose ? HIPBLAS_OP_T : HIPBLAS_OP_N);

                std::string datatype;
                io.mapRequired("DataTypeA", datatype);
                l.problem.setTypeA(string_to_hip_datatype_assert(datatype));
                io.mapRequired("DataTypeB", datatype);
                l.problem.setTypeB(string_to_hip_datatype_assert(datatype));
                io.mapRequired("DataTypeC", datatype);
                l.problem.setTypeC(string_to_hip_datatype_assert(datatype));
                io.mapRequired("DataTypeD", datatype);
                l.problem.setTypeD(string_to_hip_datatype_assert(datatype));

                std::string computetype;
                io.mapRequired("ComputeType", computetype);
                l.problem.setTypeCompute(computetype == ""
                                             ? (HIPBLAS_COMPUTE_32F)
                                             : string_to_hipblas_computetype_assert(computetype));

                // Epilogue
                std::string epilogue;
                io.mapOptional("Epilogue", epilogue);
                l.epilogue.setMode(string_to_epilogue_type_assert(epilogue));
                l.is_using_bias = is_bias_enabled(l.epilogue.getMode());
                if(l.is_using_bias)
                {
                    io.mapRequired("BiasType", datatype);
                    l.epilogue.setBiasDataType(string_to_hip_datatype_assert(datatype));
                }

                // Algo index
                io.mapOptional("AlgoIndex", l.algo_index);
            }
        };
        template <>
        struct MappingTraits<LayerConfigIO>
        {
            static void mapping(IO& io, LayerConfigIO& lc)
            {
                io.mapRequired("GeneralSettings", lc.gs);
                io.mapRequired("Layers", lc.layer);
            }
        };
    }
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " yaml-config.yaml" << std::endl;
        exit(1);
    }
    auto              inputFile = llvm::MemoryBuffer::getFile(argv[1]);
    LayerConfigIO     rv;
    llvm::yaml::Input yin((*inputFile)->getMemBufferRef());
    yin >> rv;

    uint32_t rotating           = rv.gs.rotating * 1024 * 1024;
    int32_t  cold_iters         = rv.gs.cold_iters;
    int32_t  iters              = rv.gs.iters;
    int64_t  max_workspace_size = rv.gs.max_workspace_size;

    std::vector<Layer>& layer = rv.layer;
    if(layer.size() == 0)
    {
        std::cerr << "Test is empty!" << std::endl;
        exit(1);
    }

    uint32_t totalRotatingSizeNeeded = 0;
    for(size_t i = 0; i < layer.size(); i++)
    {
        uint32_t size_c = 0, size_bias = 0;
        if(layer[i].beta != 0)
        {
            size_c = layer[i].m * layer[i].n * type2Size(layer[i].problem.getTypeC());
        }
        if(layer[i].is_using_bias)
        {
            size_bias = layer[i].m * type2Size(layer[i].problem.getTypeD());
        }
        totalRotatingSizeNeeded
            += layer[i].m * layer[i].n * type2Size(layer[i].problem.getTypeA())
               + layer[i].n * layer[i].k * type2Size(layer[i].problem.getTypeB()) + size_c
               + layer[i].m * layer[i].n * type2Size(layer[i].problem.getTypeD()) + size_bias;
    }
    // Calculating block count
    int32_t max_iters   = max(cold_iters, iters);
    int32_t block_count = max(1, min(max_iters, ceil((float)rotating / totalRotatingSizeNeeded)));
    if(rotating > 0)
    {
        std::cout << "Rotating buffer " << (float)rotating / (1024 * 1024) << " MiB. "
                  << "Needed Size: " << (float)totalRotatingSizeNeeded / (1024 * 1024) << " MiB. "
                  << "Needed block count: " << block_count << " (Capped to max iters: " << max_iters
                  << ")" << std::endl;
    }

    hipStream_t       stream;
    hipblasLtHandle_t handle;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));

    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;
    for(size_t i = 0; i < layer.size(); i++)
    {
        Layer& l = layer[i];
        l.gemms  = std::make_shared<std::vector<hipblaslt_ext::Gemm>>();
        if(l.type == Layer::TYPE::FLUSH)
        {

            continue;
        }
        l.initBlock(block_count);
        if(l.problem.getOpA() == HIPBLAS_OP_N)
            initAndCopy(&l.a, l.d_a, l.m, l.k, l.batch, l.problem.getTypeA(), block_count);
        else
            initAndCopy(&l.a, l.d_a, l.k, l.m, l.batch, l.problem.getTypeA(), block_count);
        if(l.problem.getOpB() == HIPBLAS_OP_N)
            initAndCopy(&l.b, l.d_b, l.k, l.n, l.batch, l.problem.getTypeB(), block_count);
        else
            initAndCopy(&l.b, l.d_b, l.n, l.k, l.batch, l.problem.getTypeB(), block_count);
        initNoCopy(&l.c, l.d_c, l.m, l.n, l.batch, l.problem.getTypeC(), block_count);
        initNoCopy(&l.d, l.d_d, l.m, l.n, l.batch, l.problem.getTypeD(), block_count);
        if(l.is_using_bias)
            initAndCopy(
                &l.bias, l.d_bias, 1, 1, l.batch, l.epilogue.getBiasDataType(), block_count);
        for(int b = 0; b < block_count; b++)
        {
            l.gemms->push_back(hipblaslt_ext::Gemm(handle,
                                                   l.problem.getOpA(),
                                                   l.problem.getOpB(),
                                                   l.problem.getTypeA(),
                                                   l.problem.getTypeB(),
                                                   l.problem.getTypeC(),
                                                   l.problem.getTypeD(),
                                                   l.problem.getTypeCompute()));

            hipblaslt_ext::GemmInputs inputs;
            inputs.setA(l.d_a[b]);
            inputs.setB(l.d_b[b]);
            inputs.setC(l.d_c[b]);
            inputs.setD(l.d_d[b]);
            inputs.setAlpha(&l.alpha);
            inputs.setBeta(&l.beta);
            inputs.setBias(l.d_bias[b]);
            (*l.gemms)[b].setProblem(l.m, l.n, l.k, l.batch, l.epilogue, inputs);
        }

        std::vector<hipblasLtMatmulHeuristicResult_t> tmpResult;
        size_t                                        workspaceSizeInBytes = 0;
        if(l.algo_index != -1)
        {
            std::cout << "Selected index for layer " << i << " is: " << l.algo_index << "."
                      << std::endl;

            std::vector<int> algoIndex = {l.algo_index};
            auto status = hipblaslt_ext::getAlgosFromIndex(handle, algoIndex, tmpResult);
            if(HIPBLAS_STATUS_INVALID_VALUE == status
               || (*l.gemms)[0].isAlgoSupported(tmpResult[0].algo, workspaceSizeInBytes)
                      != HIPBLAS_STATUS_SUCCESS)
            {
                std::cout << "Error invalid index for layer " << i << "." << std::endl;
                exit(1);
            }
        }
        else
        {
            tmpResult.clear();
            (*l.gemms)[0].algoGetHeuristic(1, gemmPref, tmpResult);
            if(tmpResult.size() == 0)
            {
                std::cout << "No Solution found for layer " << i << "." << std::endl;
                exit(1);
            }
            if((*l.gemms)[0].isAlgoSupported(tmpResult[0].algo, workspaceSizeInBytes)
               != HIPBLAS_STATUS_SUCCESS)
            {
                std::cout << "Error invalid index for layer " << i << "." << std::endl;
                exit(1);
            }
        }
        heuristicResults.push_back(tmpResult[0]);
        l.ws_size = workspaceSizeInBytes;
        CHECK_HIP_ERROR(hipMalloc(&l.ws, workspaceSizeInBytes));
    }

    // Get device information
    hipDeviceProp_t deviceProps;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&deviceProps, 0));
    int32_t gpu_block3 = deviceProps.multiProcessorCount * 60;

    for(int b = 0; b < block_count; b++)
        for(size_t gemmIdx = 0; gemmIdx < layer.size(); gemmIdx++)
        {
            if(layer[gemmIdx].type == Layer::TYPE::GEMM)
            {
                (*layer[gemmIdx].gemms)[b].setMaxWorkspaceBytes(layer[gemmIdx].ws_size);
                CHECK_HIPBLASLT_ERROR((*layer[gemmIdx].gemms)[b].initialize(
                    heuristicResults[gemmIdx].algo, layer[gemmIdx].ws));
            }
        }

    hipEvent_t event_gpu_time_start, event_gpu_time_end;
    CHECK_HIP_ERROR(hipEventCreate(&event_gpu_time_start));
    CHECK_HIP_ERROR(hipEventCreate(&event_gpu_time_end));

    for(int i = 0; i < cold_iters; i++)
    {
        for(size_t gemmIdx = 0; gemmIdx < layer.size(); gemmIdx++)
        {
            if(layer[gemmIdx].type == Layer::TYPE::GEMM)
                static_cast<void>((*layer[gemmIdx].gemms)[i % block_count].run(stream));
        }
    }

    CHECK_HIP_ERROR(hipEventSynchronize(event_gpu_time_start));
    CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_start, stream));
    hipGraph_t graph = NULL;
    if(rv.gs.graph_mode)
    {
        hipStreamCaptureMode mode = hipStreamCaptureModeGlobal;
        CHECK_HIP_ERROR(hipStreamBeginCapture(stream, mode));
    }

    for(int i = 0; i < iters; i++)
    {
        for(size_t gemmIdx = 0; gemmIdx < layer.size(); gemmIdx++)
        {
            switch(layer[gemmIdx].type)
            {
            case Layer::TYPE::GEMM:
                static_cast<void>((*layer[gemmIdx].gemms)[i % block_count].run(stream));
                break;
            case Layer::TYPE::FLUSH:
                hipLaunchKernelGGL(flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
                break;
            default:
                break;
            }
        }
    }

    if(rv.gs.graph_mode)
    {
        CHECK_HIP_ERROR(hipStreamEndCapture(stream, &graph));
    }
    CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_end, stream));
    CHECK_HIP_ERROR(hipEventSynchronize(event_gpu_time_end));

    if(rv.gs.graph_mode)
    {
        hipGraphExec_t graph_exec = NULL;
        CHECK_HIP_ERROR(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        CHECK_HIP_ERROR(hipEventSynchronize(event_gpu_time_start));
        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_start, stream));
        CHECK_HIP_ERROR(hipGraphLaunch(graph_exec, stream));
        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_end, stream));
        CHECK_HIP_ERROR(hipEventSynchronize(event_gpu_time_end));
        CHECK_HIP_ERROR(hipGraphExecDestroy(graph_exec));
    }
    float gpu_time_ms;
    CHECK_HIP_ERROR(hipEventElapsedTime(&gpu_time_ms, event_gpu_time_start, event_gpu_time_end));
    auto gpu_time_used = gpu_time_ms * 1000; // ms to us
    std::cout << "Time: " << gpu_time_used / iters << std::endl;

    // Print kernel info
    if(rv.gs.print_kernel_info)
    {
        std::cout << "Kernel Information:" << std::endl;
        for(size_t gemmIdx = 0; gemmIdx < layer.size(); gemmIdx++)
        {
            if(layer[gemmIdx].type == Layer::TYPE::GEMM)
            {
                auto kernelname = (*layer[gemmIdx].gemms)[0].getSolutionName();
                std::cout << "[" << gemmIdx << "] " << kernelname << std::endl;
            }
        }
    }

    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipEventDestroy(event_gpu_time_start));
    CHECK_HIP_ERROR(hipEventDestroy(event_gpu_time_end));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    if(graph)
    {
        CHECK_HIP_ERROR(hipGraphDestroy(graph));
    }
    return 0;
}
