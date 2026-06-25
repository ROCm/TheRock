// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cassert>

#include "ck/library/utility/host_tensor.hpp"

namespace ck {

std::size_t HostTensorDescriptor::GetNumOfDimension() const { return mLens.size(); }

std::size_t HostTensorDescriptor::GetElementSize() const
{
    assert(mLens.size() == mStrides.size());
    return std::accumulate(
        mLens.begin(), mLens.end(), std::size_t{1}, std::multiplies<std::size_t>());
}

std::size_t HostTensorDescriptor::GetElementSpaceSize() const
{
    std::size_t space = 1;
    for(std::size_t i = 0; i < mLens.size(); ++i)
    {
        if(mLens[i] == 0)
            continue;

        space += (mLens[i] - 1) * mStrides[i];
    }
    return space;
}

const std::vector<std::size_t>& HostTensorDescriptor::GetLengths() const { return mLens; }

const std::vector<std::size_t>& HostTensorDescriptor::GetStrides() const { return mStrides; }

std::ostream& operator<<(std::ostream& os, const HostTensorDescriptor& desc)
{
    os << "dim " << desc.GetNumOfDimension() << ", ";

    os << "lengths {";
    LogRange(os, desc.GetLengths(), ", ");
    os << "}, ";

    os << "strides {";
    LogRange(os, desc.GetStrides(), ", ");
    os << "} ";

    return os;
}

std::ostream& operator<<(std::ostream& os, HostTensorDescriptor::ChosenLayout tag)
{
    switch(tag)
    {
    case HostTensorDescriptor::ChosenLayout::Original: os << "Original"; break;
    case HostTensorDescriptor::ChosenLayout::RowMajor: os << "RowMajor"; break;
    case HostTensorDescriptor::ChosenLayout::ColumnMajor: os << "ColumnMajor"; break;
    }
    return os;
}

} // namespace ck
