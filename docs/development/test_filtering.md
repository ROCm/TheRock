# Test Filtering

`TheRock` has various stages where each stage will apply a specific test filter.

## Types of filters

- <b>smoke</b>: A "sanity check" to ensure the system is fundamentally working
  - Runs on: pull requests (if ROCm non-component related change), push to main branch
  - Characteristics: Shallow validation, focus on critical paths, component runs properly
  - Execution time: < 5 min
  - Example: pull request change to build system, main branch push change for CI

<br/>

- <b>standard</b>: The core baseline tests that ensures the most important and most commonly used functionality of the system are working
  - Runs on: pull requests, workflow dispatch, push to main branch (if ROCm component related change)
  - Characteristics: business-critical logic, covers functionality that would block users or cause major regressions, high signal-to-noise ratio
  - Execution time: < 30 min
  - Example: submodule bump in TheRock (rocm-libraries), pull request change to hipblaslt runs hipblaslt and related subproject tests

<br/>

- <b>nightly</b>: Test set that builds on top of standard tests, extending deeper test coverage
  - Runs on: nightly
  - Characteristics: deeper validation of edge cases, more expensive scenarios, more combinations of tests
  - Execution time: < 2 hours
  - Example: daily scheduled GitHub Action run

<br/>

- <b>full</b>: Test set that provides the highest level of confidence, validating a system under all conditions and edge cases
  - Runs on: weekly, pre-major release
  - Characteristics: exhaustive scenarios, extreme edge cases, aim to eliminate unknown risks
  - Execution time: 2+ hours
  - Example: pre-release test run

## Test filter implementation

For gtest executables, using `gtest_filter` is sufficient

```
./gtest-executable --gtest_filter=*smoke*
./gtest-executable --gtest_filter=*nightly*
```

For ctest, using the `GTEST_FILTER` environment variable with ctest executables will be sufficient like below:

```
SMOKE_TESTS = [
  "*smoke_tests*",
  "*basic_tests*"
]
environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    environ_vars["GTEST_FILTER"] = ":".join(SMOKE_TESTS)
```

## Additional information

- Each test filter should build on top of each other, to bring confidence to ROCm at each stage of development
- Execution time means total test time (excluding environment setup) with no sharding
- These test execution times will be enforced with GitHub Actions step timeouts, and going over the timeout will cause a CI failure
