# Test Filtering

`TheRock` has various stages where each stage will apply a specific test filter.

## Types of filters

- <b>quick</b>: A "sanity check" to ensure the system is fundamentally working

  - Runs on: pull requests (if ROCm non-component related change), push to main branch
  - Characteristics: Shallow validation, focus on critical paths, component runs properly
  - Execution time: < 5 min
  - Target: build system updates, CI system updates, non-component specific updates

> Note: This label is not intended to run on component changes, this is for build/CI changes that impacts all components and need a quick sanity check

<br/>

- <b>standard</b>: The core baseline tests that ensures the most important and most commonly used functionality of the system are working
  - Runs on: pull requests, workflow dispatch, push to main branch (if ROCm component related change)
  - Characteristics: business-critical logic, covers functionality that would block users or cause major regressions, high signal-to-noise ratio
  - Execution time: < 30 min
  - Target: component-specific core updates, submodule bumps

<br/>

- <b>comprehensive</b>: Test set that builds on top of standard tests, extending deeper test coverage
  - Runs on: nightly
  - Characteristics: deeper validation of edge cases, more expensive scenarios, more combinations of tests
  - Execution time: < 2 hours
  - Target: daily scheduled run, on-demand PR label

<br/>

- <b>full</b>: Test set that provides the highest level of confidence, validating a system under all conditions and edge cases
  - Runs on: weekly, pre-major release
  - Characteristics: exhaustive scenarios, extreme edge cases, aim to eliminate unknown risks
  - Execution time: 2+ hours
  - Target: pre-release testing, weekly scheduled run, on-demand PR label

## Test filter implementation

Test filter implementation is done with CTest.
Whatever be the underlying test framework - say gtest, pytest etc - a ctest wrapper will be created over it exposing the capability to run each test category using ctest labels.

To do this the implementation uses a test_categories.yaml file which provides the template to add/exclude the tests to be run for each category, which has to be updated by the component teams. We can add/exclude tests based on the gpu model and OS where the tests are run.

TheRock CI uses the environment variables `TEST_TYPE` to specify the test category and `AMDGPU_FAMILIES` for gpu.

A sample ctest command for a `quick` test run on `gfx110X` will look like

```
ctest -L quick -L ex_gpu_gfx110X
```

More information on implementation and integration is available in the below links:

https://github.com/ROCm/rocm-libraries/blob/develop/shared/ctest/README.md
https://github.com/ROCm/TheRock/blob/main/build_tools/github_actions/test_executable_scripts/README.md

## Additional information

- Each test filter should build on top of each other, to bring confidence to ROCm at each stage of development
- Execution time means total test time (excluding environment setup) with no sharding
- These test execution times will be enforced with GitHub Actions step timeouts, and going over the timeout will cause a CI failure
