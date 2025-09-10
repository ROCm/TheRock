# Test Debugging

When tests fail on a specific platform or GPU, see these instructions for
suggestions on how to debug and file useful bug reports.

## Debugging when tests fail on GitHub workflows

When tests fail on a GitHub Actions CI workflow, developers can follow the
[Test Environment Reproduction](./test_environment_reproduction.md) guide to
run tests locally, at which point they can enable additional logging and perform
other debugging. See also (TODO: link) for instructions on how to bootstrap
your local build from CI artifacts.

TODO: some way to opt in for additional logging as parpt of a CI workflow? We
could add a `debug` boolean option to workflow_dispatch for try-jobs.

## Per-project debugging guides

Many ROCm subprojects control logging and additional debugging via environment
variables. Here is a collection of some of the subproject documentation.

### HIP debugging

https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/debugging.html

```bash
# Enable debug HIP logging
export AMD_LOG_LEVEL=4
```

### hipBLASLt debugging

https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/logging-heuristics.html

```bash
export HIPBLASLT_LOG_LEVEL=4
```

### MIOpen debugging

https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/debug-log.html

```bash
# Enable layer-by-layer and command line MIOpen debug logging
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_ENABLE_LOGGING_CMD=1
export MIOPEN_LOG_LEVEL=6
```

### rocBLAS debugging

https://rocm.docs.amd.com/projects/rocBLAS/en/latest/how-to/logging-in-rocblas.html
