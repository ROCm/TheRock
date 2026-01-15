# Test Filtering

`TheRock` has various stages where each stage will apply a specific test filter.

## Types of filters

- <b>smoke</b>: A "sanity check" to ensure the system is fundamentally working
  - Runs on: pull requests (if ROCm non-component related change)
  - Characteristics: Shallow validation, focus on critical paths, component runs properly
  - Execution time: < 5 min

<br/>

- <b>standard</b>: The core baseline tests that ensures the most important and most commonly used functionality of the system are working
  - Runs on: pull requests, workflow dispatch, push to main branch (if ROCm component related change)
  - Characteristics: business-critical logic, covers functionality that would block users or cause major regressions, high signal-to-noise ratio
  - Execution time: < 30 min

<br/>

- <b>nightly</b>: Test set that builds on top of standard tests, extending deeper test coverage
  - Runs on: nightly
  - Characteristics: deeper validation of edge cases, more expensive scenarios, more combinations of tests
  - Execution time: < 2 hours

<br/>

- <b>full</b>: Test set that provides the highest level of confidence, validating a system under all conditions and edge cases
  - Runs on: weekly, pre-major release
  - Characteristics: exhaustive scenarios, extreme edge cases, aim to eliminate unknown risks
  - Execution time: 2+ hours

<br/>

Each test filter should build on top of each other, to bring confidence to ROCm at each stage of development
