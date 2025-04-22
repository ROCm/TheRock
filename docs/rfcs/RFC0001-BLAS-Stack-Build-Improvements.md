---
author: David Dixon (ellosel)
created: 2025-04-10
modified: 2025-04-10
status: draft
---

# BLAS libraries monorepo and build system improvements

## Overview

The Math Libraries stack is prohibitively expensive to build, which introduces considerable risk to ROCm. One way to manage growing build times is to expose more parallelization in the build by composing many projects into a monorepo. However, the current state of CMake support in many Math Libraries components is not compatible with a monorepo model.

Our goals are to (1) **improve the CMake support, enabling project composability necessary** to (2) **move the BLAS libraries to a monorepo structure and gain the associated benefits** (see appendix). In doing so, we will reduce risks in the form of defects and long build times and remove unnecessary complexity that will **unlock developer productivity and remove barriers to external contributions to ROCm**.

We will use a multi-layered approach with layer (1) including component-level build system improvements that are collected one component at a time into layer (2) which is a monorepo representing the entire BLAS library stack. This will allow us to address the deficiencies in the build system of each component in isolation while incrementally integrating each component into the monorepo to demonstrate correctness before moving on. Once all components are integrated and component teams are ready to move to the new project, the process will be quick and low risk.

## Alternatives considered

The notable alternative to a monorepo approach is a project-based approach analogous to the current structure of ROCm. The project-based approach allows projects that should have high cohesion to diverge in style, structure, design and so on. This approach led to the deficiencies we face today:

- Lack of uniformity across projects
- Project specific workarounds that prevent composition
- Artificial boundaries between projects that introduce duplication and silos

## Component-level changes

In this section, we describe why component-level build system modifications are required, why we recommend additional changes to align with best-practices, and how we will accommodate these changes while the components are under active development.

Component-level build system modifications are necessary to build the monorepo because in some components:

- The build system is tightly coupled to the build environment, precluding hermetic builds.
- Hard-coded variables diminish usability and provide a mechanism to silently build invalid configurations.
- Changing unrelated functionality may require rebuilding the world rather than a subset of the relevant binaries.

In parallel to correcting the deficiencies above, we plan to align each project with what are considered best practices by the CMake community and the tensile infrastructure team. This has many advantages including:

- Improved maintainability
- Ease of use and extension

and will require a relatively small level of effort. For examples of best practices and recommendations, see the appendix.

To ensure routine development is not impacted while the component-level changes are developed, we will provide a new source tree that parallels the current tree. The new source tree will include only the CMakeLists.txt files necessary to support the new build system and will otherwise refer to the current source tree for the actual source files. This will prevent disruptions to on-going development as the monorepo comes online. The following components will be migrated to the monorepo in the order listed below:

1. rocJenkins (optional)
1. hipBLAS-common
1. rocRoller (optional)
1. Oragami (optional)
1. hipBLASLt
1. Tensile
1. rocBLAS
1. rocSparse
1. hipSparse
1. rocSolver
1. hipSparseLt
1. hipSolver
1. hipBLAS

## Building the monorepo

As the component-wise development is completed, the individual components will be migrated into a pre-existing monorepo. Automated builds and tests will run on the monorepo continuously to ensure correctness and to collect performance data. A POC from each Math library team will be requested as a beta tester as necessary.

## Regression plan

Given that the plan involves maintaining two separate source trees, we will need sufficient testing to detect regressions. To do so, we will integrate the monorepo into TheRock and make use of the existing automated testing for the new build system while continuing to test the existing build system through MathCI. In addition, we will work directly with the MathCI team as they begin work to adopt the monorepo in rocJenkins.

## Communication

Prior to starting the work, we will create a blas-libraries mailing list to solicit beta testers and provide routine updates on the status of the project. We request that one of our team members is involved in any PR that impacts the existing build system.

# Appendix

## Benefits of hermeticity (taken from bazel docs)

- **Speed**: The output of an action can be cached, and the action need not be run again unless inputs change.
- **Parallel execution**: For given input and output, the build system can construct a graph of all actions to calculate efficient and parallel execution. The build system loads the rules and calculates an action graph and hash inputs to look up in the cache.
- **Multiple builds**: You can build multiple hermetic builds on the same machine, each build using different tools and versions.
- **Reproducibility**: Hermetic builds are good for troubleshooting because you know the exact conditions that produced the build.

## Benefits of a monorepo (taken from Kali Linux Tutorials)

- **Code sharing and reusability**: Monorepos encourage the reuse of shared libraries, components, and utilities across projects. Tools like Nx and Lerna facilitate the creation and management of shared modules, ensuring consistency across applications while reducing duplication.
- **Simplified dependency management**: Monorepo tools centralize dependency management, allowing developers to maintain uniform versions of libraries across all projects. For instance, updating a shared library propagates changes to all dependent projects automatically, reducing version conflicts.
- **Atomic changes**: Monorepos enable developers to make atomic changes—modifying shared components or APIs and updating all dependent projects in a single commit. This eliminates the need for coordinating changes across multiple repositories.
- **Task orchestration and build optimization**: Tools like Bazel, Nx, and Turborepo optimize builds by analyzing dependencies and running only necessary tasks. Features like local computation caching and distributed task execution significantly improve build times for large-scale monorepos.
- **Continuous integration (CI)**: CI pipelines tailored for monorepos can selectively build and test only the modules affected by recent changes. This reduces overhead compared to rebuilding the entire codebase.
- **Standardization across projects**: By housing all projects in one repository, monorepos enforce consistent coding standards, linting rules, and configurations across teams. This fosters better collaboration and code quality.

## Recommendations for best practices

- **General**

  - The build system should not be responsible for build environment setup
  - The build system should not be responsible for managing 1st party dependencies
  - Where possible, avoid 3rd party dependency management in the build system
  - Don't change the behavior of the configuration based on whether a package is found
  - Avoid FetchContent and git submodules or at least look for a system/local install first
  - Don’t leak build system implementation to downstream consumers
  - Do not provide all possible configurations and put unused configurations like LEGACY_HIPBLAS_DIRECT on a path to deprecation
  - Projects should be designed to be composable by a higher-level CMakeLists.txt

- **Style and structure**

  - Structure CMakeLists.txt consistently. For example, the root file could have the following structure:
    - options/cache variables
    - includes
    - find_package calls
    - declare targets
    - Testing support
    - add_subdirectory calls
    - Install support
    - CPack support
  - Use common indentation and whitespace throughout projects
  - Remove unhelpful comments
  - Consider pitchfork layout or any standardized, recognizable layout appropriate for each project and the associated language

- **Customization**

  - Prefer native CMake functionality over in-house functions/macros
  - Common usage patterns such as adding flags for code coverage, sanitizers, or docs generation, can be implemented as a CMake target, separated into a \*.cmake file and added to a project wide cmake directory for reuse via include

- **CMake usage**

  - Find packages in highest level CMakeLists.txt
  - Mark packages as required and fail loudly if they aren't found
  - Use target\_ functions rather than global functions such as include_directories
  - Each project should declare options, targets, aliases, and dependencies in their highest-level CMakeLists.txt
  - Use target_sources to associate source files with targets in subdirectories
  - Use build and install interface correctly/consistently
  - No hard-coded options
  - Avoid proliferation of customization points that could result in an invalid configuration
  - Do not set variables like CMAKE_CXX_COMPILER in project - prefer a toolchain file
  - Consider avoiding usage of option function and prefer setting a cache variable as option has surprising behavior
  - Prefer robust ways of integrating build targets over execute_process, custom_commands, and so on that trigger other build scripts.
  - Do not change the semantics of the built-in build types; instead, create custom build types and associate CMAKE_CXX_FLAGS etc with custom build types.
  - Prefer CMake find modules and if you must roll your own, name it "Find\<project>.cmake" and provide a robust implementation (see examples from CMake project on github)
  - Use targets to propagate information
  - Don't emit messages that aren't essential - they clutter the configuration/build output
  - Use built-in variables such as BUILD_TESTING and BUILD_SHARED_LIBS
  - Don't explicitly set the library type. It defaults to STATIC. It can't be configured if it is explicitly set.
  - Don't include CMakeLists.txt files; use add_subdirectory
  - Prefer native CMake variables over variables such as ROCBLASLT_SHARED_LIBS
