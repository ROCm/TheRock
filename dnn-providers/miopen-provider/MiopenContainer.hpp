// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <set>

namespace miopen_legacy_plugin
{

class EngineManager;

/*
 * Container class to manage the intantiation and ownership of all MIOpen plan builders and engines.
 * The class designs use dependency injection to get the components they need in order to function.
 * This makes it easier to test and maintain the code as you can swap out implementations.
 *
 * The construction sequence should contain no logic other than the creation of various classes.
 * If logic is needed, it should be placed in a separate function that can be called after the
 * container has finished constructing all its components.
 */
class MiopenContainer
{
public:
    MiopenContainer();
    ~MiopenContainer();

    EngineManager& getEngineManager();

private:
    std::unique_ptr<EngineManager> _engineManager;
};

}
