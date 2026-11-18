# Rocddi runtime components

Linux builds can enable `rocm-systems/runtimes/` with
`THEROCK_ENABLE_ROCDDI_RUNTIMES=ON`. The option defaults to `OFF` and requires
TheRock's BASE and COMPILER features. It is independent of
`THEROCK_ENABLE_CORE_RUNTIME`.

Prepare Rust, Cargo and dependency sources as described in the selected
`rocm-systems/runtimes/` revision. The build forwards `ROCM_RUNTIMES_CARGO`,
`ROCM_RUNTIMES_RUSTC`, `ROCM_RUNTIMES_LLD`, `ROCM_RUNTIMES_CARGO_HOME`,
`ROCM_RUNTIMES_CARGO_CONFIG` and `ROCM_RUNTIMES_CARGO_JOBS` when supplied.
Relative Cargo configuration paths are anchored to the initial CMake invocation
directory. Clearing configuration or jobs removes the previous child setting.
An absent or explicitly empty Cargo home selects the upstream build's isolated
`cargo-home` directory. Rerun outer CMake after changing `RUSTFLAGS`,
`CARGO_ENCODED_RUSTFLAGS` or `PATH`; upstream captures them during configuration.
The runtime source follows TheRock's rocm-systems revision or its existing
`THEROCK_ROCM_SYSTEMS_SOURCE_DIR` override.

Add these options to an otherwise configured TheRock Ninja build:

```sh
cmake -S . -B build -DTHEROCK_ENABLE_ROCDDI_RUNTIMES=ON -DTHEROCK_BUILD_TESTING=ON
cmake --build build --target rocm-runtimes build-test-rocm-runtimes
```

Upstream CMake and Cargo own compilation and host-test registration. The test
target runs upstream CTest tests labeled `host` with upstream's selected
toolchain and offline environment. Libraries and native test executables remain
under `build/core/rocm-runtimes/build/`; this
integration supplies no SDK install or artifact payload.
