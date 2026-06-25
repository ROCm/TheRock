// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>

#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "MiopenConvBwdPlan.hpp"
#include "MiopenUtils.hpp"

namespace miopen_legacy_plugin
{

ConvBwdParams::ConvBwdParams(
    const hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _spatialDimCount(miopen_utils::getSpatialDimCount(
          miopen_utils::findTensorAttributes(tensorMap, attributes.dx_tensor_uid())))
    , _dx(miopen_utils::createTensor(tensorMap, attributes.dx_tensor_uid()))
    , _w(miopen_utils::createTensor(tensorMap, attributes.w_tensor_uid()))
    , _dy(miopen_utils::createTensor(tensorMap, attributes.dy_tensor_uid()))
{
    const auto& attrDX = miopen_utils::findTensorAttributes(tensorMap, _dx.uid());
    const auto& attrW = miopen_utils::findTensorAttributes(tensorMap, _w.uid());
    const auto& attrDY = miopen_utils::findTensorAttributes(tensorMap, _dy.uid());

    const auto inputDims
        = hipdnn_data_sdk::utilities::convertFlatBufferVectorToStdVector(attrDX.dims());
    const auto weightDims
        = hipdnn_data_sdk::utilities::convertFlatBufferVectorToStdVector(attrW.dims());
    const auto groupCount = hipdnn_data_sdk::utilities::calculateGroupCount(inputDims, weightDims);

    _conv = MiopenConvDescriptor(_spatialDimCount, attributes, static_cast<int>(groupCount));

    _tensorsValid = (!attrDX.virtual_() && !attrW.virtual_() && !attrDY.virtual_());
}

const MiopenTensor& ConvBwdParams::dx() const
{
    return _dx;
}

const MiopenTensor& ConvBwdParams::w() const
{
    return _w;
}

const MiopenTensor& ConvBwdParams::dy() const
{
    return _dy;
}

const MiopenConvDescriptor& ConvBwdParams::conv() const
{
    return _conv;
}

bool ConvBwdParams::validTensors() const
{
    return _tensorsValid;
}

ConvBwdPlan::ConvBwdPlan(const HipdnnEnginePluginHandle& handle, ConvBwdParams&& params)
    : _params(std::move(params))
{
    // MIOpen Find 2.0 API
    miopenProblem_t problem;
    THROW_ON_MIOPEN_FAILURE(miopenCreateConvProblem(
        &problem, _params.conv().convDescriptor(), miopenProblemDirectionBackward));
    hipdnn_data_sdk::utilities::ScopedResource problemRes(
        problem, [](miopenProblem_t p) { std::ignore = miopenDestroyProblem(p); });

    THROW_ON_MIOPEN_FAILURE(miopenSetProblemTensorDescriptor(
        problem, miopenTensorConvolutionX, _params.dx().tensorDescriptor()));
    THROW_ON_MIOPEN_FAILURE(miopenSetProblemTensorDescriptor(
        problem, miopenTensorConvolutionW, _params.w().tensorDescriptor()));
    THROW_ON_MIOPEN_FAILURE(miopenSetProblemTensorDescriptor(
        problem, miopenTensorConvolutionY, _params.dy().tensorDescriptor()));

    size_t numSolutions;
    miopenSolution_t solution = nullptr;
    // Requesting only the best solution
    THROW_ON_MIOPEN_FAILURE(
        miopenFindSolutions(handle.miopenHandle, problem, nullptr, &solution, &numSolutions, 1));

    if(solution != nullptr)
    {
        _solution = hipdnn_data_sdk::utilities::ScopedResource<miopenSolution_t>(
            solution, [](miopenSolution_t s) {
                auto status = miopenDestroySolution(s);
                if(status != miopenStatusSuccess)
                {
                    HIPDNN_LOG_ERROR("miopenDestroySolution failed in ConvBwdPlan destructor");
                }
            });
    }

    if(numSolutions != 1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "miopenFindSolutions returned no solutions");
    }

    THROW_ON_MIOPEN_FAILURE(miopenGetSolutionWorkspaceSize(_solution.get(), &_workspaceSize));
}

size_t ConvBwdPlan::getWorkspaceSize([[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    return _workspaceSize;
}

void ConvBwdPlan::execute(const HipdnnEnginePluginHandle& handle,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          uint32_t numDeviceBuffers,
                          void* workspace) const
{
    auto dxDesc = _params.dx().tensorDescriptor();
    auto wDesc = _params.w().tensorDescriptor();
    auto dyDesc = _params.dy().tensorDescriptor();

    auto xBuffer
        = miopen_utils::findDeviceBuffer(_params.dx().uid(), deviceBuffers, numDeviceBuffers);
    auto wBuffer
        = miopen_utils::findDeviceBuffer(_params.w().uid(), deviceBuffers, numDeviceBuffers);
    auto yBuffer
        = miopen_utils::findDeviceBuffer(_params.dy().uid(), deviceBuffers, numDeviceBuffers);

    std::array<miopenTensorArgument_t, 3> tensors
        = {miopenTensorArgument_t{miopenTensorConvolutionX, &dxDesc, xBuffer.ptr},
           miopenTensorArgument_t{miopenTensorConvolutionW, &wDesc, wBuffer.ptr},
           miopenTensorArgument_t{miopenTensorConvolutionY, &dyDesc, yBuffer.ptr}};

    size_t workspaceSize = 0;
    if(workspace != nullptr)
    {
        // Assume the provided workspace is large enough
        workspaceSize = _workspaceSize;
    }

    THROW_ON_MIOPEN_FAILURE(miopenRunSolution(handle.miopenHandle,
                                              _solution.get(),
                                              tensors.size(),
                                              tensors.data(),
                                              workspace,
                                              workspaceSize));
}

} // namespace miopen_legacy_plugin
