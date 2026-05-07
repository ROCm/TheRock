// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <type_traits>
#include <memory>

#include "ck/utility/functional2.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <typename BaseOp, typename NewOpInstances>
void add_device_operation_instances(std::vector<std::unique_ptr<BaseOp>>& op_instances,
                                    const NewOpInstances& new_op_instances)
{
    ck::static_for<0, std::tuple_size_v<NewOpInstances>, 1>{}([&](auto i) {
        const auto new_op_instance = std::get<i>(new_op_instances);

        using NewOpInstance = remove_cvref_t<decltype(new_op_instance)>;
        if constexpr(std::is_same_v<NewOpInstance, std::nullptr_t>)
        {
            return; // We can use nullptr_t to enable trailing comma
        }
        else
        {
            static_assert(std::is_base_of_v<BaseOp, NewOpInstance>,
                          "wrong! NewOpInstance should be derived from BaseOp");
            op_instances.push_back(std::make_unique<NewOpInstance>(new_op_instance));
        }
    });
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
