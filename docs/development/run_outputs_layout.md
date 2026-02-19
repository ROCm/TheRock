# CI Run Outputs Layout

This document describes the directory layout for outputs produced by TheRock's
CI workflow runs, and the Python API for computing paths to those outputs.

## Overview

Every CI workflow run produces a set of outputs (build artifacts, logs,
manifests, python packages) that are uploaded to S3. The
`_therock_utils.run_outputs` module provides a single source of truth for
computing paths to these outputs, replacing ad-hoc f-string construction
scattered across the codebase.

## S3 Layout

All outputs for a given run live under a common prefix:

```
s3://{bucket}/{external_repo}{run_id}-{platform}/
```

| Component       | Example                   | Description                                       |
| --------------- | ------------------------- | ------------------------------------------------- |
| `bucket`        | `therock-ci-artifacts`    | Selected based on repo, fork status, release type |
| `external_repo` | `""` or `"Fork-TheRock/"` | Non-empty for forks and non-TheRock repos         |
| `run_id`        | `12345678901`             | GitHub Actions workflow run ID                    |
| `platform`      | `linux` or `windows`      | Build platform                                    |

### Directory structure

```
{prefix}/
    {artifact_name}_{component}_{target_family}.tar.xz
    {artifact_name}_{component}_{target_family}.tar.xz.sha256sum
    index-{artifact_group}.html

    logs/{artifact_group}/
        build.log
        ninja_logs.tar.gz
        build_observability.html
        index.html
        comp-summary.html
        comp-summary.md

    manifests/{artifact_group}/
        therock_manifest.json

    python/{artifact_group}/
        *.whl
        *.tar.gz
```

### Bucket selection

The bucket is determined by `_retrieve_bucket_info()` based on:

- **`ROCm/TheRock` (main repo):** `therock-ci-artifacts`
- **Fork PRs:** `therock-ci-artifacts-external`
- **Nightly releases:** `therock-nightly-artifacts`
- **Stable releases:** `therock-release-artifacts`
- **Internal releases:** `therock-artifacts-internal`
- **Pre-cutover runs (before 2025-11-11):** `therock-artifacts` / `therock-artifacts-external`

## Python API

### OutputLocation

A frozen dataclass representing a single file or directory in the layout.

```python
from _therock_utils.run_outputs import OutputLocation

loc = OutputLocation(
    bucket="therock-ci-artifacts", relative_path="12345-linux/file.tar.xz"
)
loc.s3_uri  # "s3://therock-ci-artifacts/12345-linux/file.tar.xz"
loc.https_url  # "https://therock-ci-artifacts.s3.amazonaws.com/12345-linux/file.tar.xz"
loc.local_path(Path("/tmp/staging"))  # Path("/tmp/staging/12345-linux/file.tar.xz")
```

### RunOutputRoot

A frozen dataclass that computes `OutputLocation` for every output type.

```python
from _therock_utils.run_outputs import RunOutputRoot

# From CI environment (calls GitHub API for bucket selection)
root = RunOutputRoot.from_workflow_run(run_id="12345", platform="linux")

# For local development (no API calls)
root = RunOutputRoot.for_local(run_id="local", platform="linux")

# Location methods — each returns an OutputLocation
root.artifact("blas_lib_gfx94X.tar.xz")
root.artifact_index("gfx94X-dcgpu")
root.log_dir("gfx94X-dcgpu")
root.log_file("gfx94X-dcgpu", "build.log")
root.log_index("gfx94X-dcgpu")
root.build_observability("gfx94X-dcgpu")
root.manifest("gfx94X-dcgpu")
root.python_packages("gfx110X-all")
```

### Adding new output types

To add a new output type:

1. Add a method to `RunOutputRoot` that returns `OutputLocation`
1. Add tests to `build_tools/tests/run_outputs_test.py`
1. Update this document

## Consumers

| File                           | Uses                                                                  |
| ------------------------------ | --------------------------------------------------------------------- |
| `artifact_backend.py`          | `RunOutputRoot` for backend construction                              |
| `artifact_manager.py`          | Via `create_backend_from_env()`                                       |
| `post_build_upload.py`         | `RunOutputRoot.from_workflow_run()` for upload paths and summary URLs |
| `upload_test_report_script.py` | `RunOutputRoot.from_workflow_run()` for S3 base URI                   |
