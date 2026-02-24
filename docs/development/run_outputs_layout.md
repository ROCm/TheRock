# CI Run Outputs Layout

This document describes the directory layout for outputs produced by TheRock's
CI workflow runs, and the Python modules for computing paths, uploading, and
downloading those outputs.

## Overview

Every CI workflow run produces a set of outputs (build artifacts, logs,
manifests, python packages) that are uploaded to S3. Three modules in
`_therock_utils` handle the path computation and I/O:

| Module             | Role                      | Key types                                                |
| ------------------ | ------------------------- | -------------------------------------------------------- |
| `run_outputs`      | Path computation (no I/O) | `RunOutputRoot`, `OutputLocation`                        |
| `upload_backend`   | Upload I/O (write)        | `UploadBackend`, `S3UploadBackend`, `LocalUploadBackend` |
| `artifact_backend` | Download I/O (read)       | `ArtifactBackend`, `S3Backend`, `LocalDirectoryBackend`  |

`OutputLocation` is the bridge between path computation and I/O.
`RunOutputRoot` produces `OutputLocation` instances; backends consume them.

```
RunOutputRoot ──produces──> OutputLocation ──consumed by──> UploadBackend
                                                            ArtifactBackend
```

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
        build_observability.html          (when generated)
        index.html
        therock-build-prof/               (resource profiling subdirectory)
            comp-summary.html
            comp-summary.md
        comp-summary.html                 (flattened copy for direct linking)
        comp-summary.md                   (flattened copy for direct linking)

    manifests/{artifact_group}/
        therock_manifest.json

    python/{artifact_group}/
        *.whl
        *.tar.gz
        index.html
```

The `comp-summary.*` files appear both in the `therock-build-prof/` subdirectory
(uploaded as part of the recursive directory upload) and at the log root
(uploaded explicitly for direct linking).

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

# Inside a CI workflow (env vars provide bucket info, no API call)
root = RunOutputRoot.from_workflow_run(run_id="12345", platform="linux")

# Fetching artifacts from another run (API call for fork/cutover detection)
root = RunOutputRoot.from_workflow_run(
    run_id="12345", platform="linux", lookup_workflow_run=True
)

# For local development (no API calls, no env vars needed)
root = RunOutputRoot.for_local(run_id="local", platform="linux")

# Location methods — each returns an OutputLocation
root.root()  # run output root directory
root.artifact("blas_lib_gfx94X.tar.xz")
root.artifact_index("gfx94X-dcgpu")
root.log_dir("gfx94X-dcgpu")
root.log_file("gfx94X-dcgpu", "build.log")
root.log_index("gfx94X-dcgpu")
root.build_observability("gfx94X-dcgpu")
root.manifest_dir("gfx94X-dcgpu")
root.manifest("gfx94X-dcgpu")
root.python_packages("gfx110X-all")
```

The `lookup_workflow_run` parameter controls whether `from_workflow_run()` calls
the GitHub API to fetch workflow run metadata (for fork detection and bucket
cutover dating). Most callers running inside their own CI workflow do not need
this — environment variables (`GITHUB_REPOSITORY`, `IS_PR_FROM_FORK`) suffice.
Set `lookup_workflow_run=True` when looking up another repository's workflow
run, e.g. when fetching artifacts.

### UploadBackend

An abstract base class for uploading files to S3 or a local directory.
Use `create_upload_backend()` to get the right implementation.

```python
from _therock_utils.upload_backend import create_upload_backend

backend = create_upload_backend()  # S3 (default)
backend = create_upload_backend(staging_dir=Path("/tmp/out"))  # local directory
backend = create_upload_backend(dry_run=True)  # print only

backend.upload_file(source_path, dest_location)
backend.upload_directory(source_dir, dest_location, include=["*.tar.xz*"])
```

Content-type is inferred from file extension — callers don't need to specify it.

### Adding new output types

To add a new output type:

1. Add a method to `RunOutputRoot` that returns `OutputLocation`
1. Add tests to `build_tools/tests/run_outputs_test.py`
1. Update this document

## Consumers

### Upload scripts

| File                           | Uses                                                                 |
| ------------------------------ | -------------------------------------------------------------------- |
| `post_build_upload.py`         | `RunOutputRoot` + `UploadBackend` for artifacts, logs, manifests     |
| `upload_python_packages.py`    | `RunOutputRoot` + `UploadBackend` for Python wheels and index        |
| `upload_pytorch_manifest.py`   | `RunOutputRoot` + `UploadBackend` for PyTorch manifests              |
| `upload_test_report_script.py` | `RunOutputRoot` for S3 base URI (upload not yet migrated to backend) |

### Download scripts

| File                           | Uses                                                                      |
| ------------------------------ | ------------------------------------------------------------------------- |
| `fetch_artifacts.py`           | `RunOutputRoot.from_workflow_run(lookup_workflow_run=True)` + `S3Backend` |
| `find_artifacts_for_commit.py` | `RunOutputRoot.from_workflow_run(workflow_run=...)` for bucket/prefix     |
| `artifact_backend.py`          | `RunOutputRoot` for `S3Backend` construction                              |
| `artifact_manager.py`          | Via `create_backend_from_env()`                                           |
