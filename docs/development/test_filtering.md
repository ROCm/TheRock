# Test Filtering

`TheRock` has various stages where each stage will apply a specific test filter.

> Note: Execution time means total test time (excluding environment setup) on a single shard (no parallelization)

## Types of filters

- <b>quick</b>: A "sanity check" to ensure the system is fundamentally working

  - Runs on: pull requests (non-component changes), push to main branch, dev releases
  - Characteristics: Shallow validation, focus on critical paths, component runs properly
  - Execution time: < 5 min
  - Target: build system updates, CI system updates, non-component specific updates

> Note: This label is not intended to run on component changes, this is for build/CI changes that impacts all components and need a quick sanity check

<br/>

- <b>standard</b>: The core baseline tests that ensures the most important and most commonly used functionality of the system are working
  - Runs on: pull requests (component changes), push to main branch (component changes)
  - Characteristics: business-critical logic, covers functionality that would block users or cause major regressions, high signal-to-noise ratio
  - Execution time: < 30 min
  - Target: component-specific core updates, submodule bumps

<br/>

- <b>comprehensive</b>: Test set that builds on top of standard tests, extending deeper test coverage
  - Runs on: nightly releases, scheduled CI
  - Characteristics: deeper validation of edge cases, more expensive scenarios, more combinations of tests
  - Execution time: < 2 hours
  - Target: nightly release validation, daily scheduled run, on-demand PR label

<br/>

- <b>full</b>: Test set that provides the highest level of confidence, validating a system under all conditions and edge cases
  - Runs on: prerelease builds, submodule updates, on-demand
  - Characteristics: exhaustive scenarios, extreme edge cases, aim to eliminate unknown risks
  - Execution time: 2+ hours
  - Target: pre-release testing, submodule bumps, on-demand PR label (`test_filter:full`)

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
- These test execution times will be enforced with GitHub Actions step timeouts, and going over the timeout will cause a CI failure

## Timeout enforcement

The execution times above are enforced, not advisory. They live in
`TEST_FILTER_BUDGET_MINUTES` in
[`build_tools/github_actions/fetch_test_configurations.py`](../../build_tools/github_actions/fetch_test_configurations.py),
which applies them as the `timeout-minutes` of the `Test` step in
`test_component.yml` -- the step that runs only the test script, so setup time
is excluded exactly as the definition above states.

`standard` and `comprehensive` are enforced at 1.5x their documented time (45
and 180 minutes) so that a slow runner or a flaky retry fails the test rather
than the clock. `quick` is held exact, since the point of the filter is that it
stays immediate.

Because a budget covers the whole suite run serially on one shard, a component
split across N shards gets `budget / N` per shard. A `standard` suite on six
shards has 45/6 = 8 minutes per shard, not 45. Shard counts are already
per-platform, so a component sharded differently on Linux and Windows gets a
different per-shard timeout on each.

A component's own `timeout_minutes` in `test_matrix` still applies as a ceiling,
so a suite that declares less than its budget keeps the lower value. The budget
can only tighten it. `full` is unbudgeted -- the filter is defined as "2+ hours"
with no upper bound -- so `timeout_minutes` is used verbatim there.

There is **no per-component override**. A suite that cannot finish inside its
budget needs to be made faster, sharded further, or moved to a later filter.
