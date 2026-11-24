# Test Debugging

When tests fail on a specific platform or GPU, see these instructions for
suggestions on how to debug and file useful bug reports.

## Debugging when tests fail on GitHub workflows

When tests fail on a GitHub Actions CI workflow, developers can follow the
[Test Environment Reproduction](./test_environment_reproduction.md) guide to
run tests locally, at which point they can enable additional logging and perform
other debugging. See also (TODO: link) for instructions on how to bootstrap
your local build from CI artifacts.

### Re-running failed test jobs with debug logging

Test jobs can collect additional logging without a rebuild by re-running them
from the GitHub Actions UI with the "Enable debug logging" checkbox selected
(see
[Re-running workflows and jobs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs)).
The re-run jobs test against the artifacts that the original run already built.

When debug logging is enabled, GitHub sets `runner.debug` to `1` and the test
workflows
([`test_component.yml`](/.github/workflows/test_component.yml) and
[`test_code_coverage_component.yml`](/.github/workflows/test_code_coverage_component.yml))
set `THEROCK_CI_DEBUG=1`.
[`test_runner.py`](/build_tools/github_actions/test_executable_scripts/test_runner.py)
then turns on these debug options:

| Environment variable | Effect                                                                                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ROCM_KPACK_DEBUG=1` | kpack loader diagnostics. Enabled by default, but components with verbose test logs opt out with `"rocm_kpack_debug": "0"` in their test matrix entry. |

`THEROCK_CI_DEBUG` overrides these per-component opt-outs. Future CI debug
options can key off the same flag.

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
