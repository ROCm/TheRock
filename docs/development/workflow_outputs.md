# CI Workflow Outputs Layout

This document describes the directory layout for outputs produced by TheRock's
CI workflow runs, and the Python modules for computing paths, uploading, and
downloading those outputs.

## Overview

Every CI workflow run produces a set of outputs (build artifacts, logs,
manifests, python packages) that are uploaded to S3. Three modules in
`_therock_utils` handle the path computation and I/O:

| Module             | Role                         | Key types                                                   |
| ------------------ | ---------------------------- | ----------------------------------------------------------- |
| `storage_location` | Backend-agnostic location    | `StorageLocation`                                           |
| `workflow_outputs` | CI path computation (no I/O) | `WorkflowOutputRoot`                                        |
| `storage_backend`  | Upload I/O (write)           | `StorageBackend`, `S3StorageBackend`, `LocalStorageBackend` |
| `artifact_backend` | Download I/O (read)          | `ArtifactBackend`, `S3Backend`, `LocalDirectoryBackend`     |

`StorageLocation` is the bridge between path computation and I/O.
`WorkflowOutputRoot` produces `StorageLocation` instances; backends consume them.

```
WorkflowOutputRoot ──produces──> StorageLocation ──consumed by──> StorageBackend
                                                            ArtifactBackend
```

## S3 Layout

All outputs for a given run live under a common prefix:

```
s3://{bucket}/{external_repo}{run_id}-{platform}/
```

| Component       | Example                         | Description                                                           |
| --------------- | ------------------------------- | --------------------------------------------------------------------- |
| `bucket`        | `therock-ci-artifacts`          | Selected based on repo, fork status, release type                     |
| `external_repo` | `""` or `"githubuser-TheRock/"` | Non-empty for forks and non-TheRock repos (format: `{owner}-{repo}/`) |
| `run_id`        | `12345678901`                   | GitHub Actions workflow run ID                                        |
| `platform`      | `linux` or `windows`            | Build platform                                                        |

### Directory structure

There are two CI pipeline architectures with different log layouts:

- **Single-stage CI** (`ci.yml`) — one monolithic build per artifact group.
  Logs from all subprojects land in a single flat directory.
- **Multi-arch CI** (`multi_arch_ci.yml`) — the build is split into stages
  (foundation, compiler-runtime, math-libs, etc.). Each stage runs as a
  separate job and uploads its own logs, organized by stage name and GPU family.

`artifact_group` is the CI matrix variant, composed of a target family plus an
optional variant suffix (e.g., `gfx94X-dcgpu`, `gfx94X-dcgpu-asan`). It is
used as the subdirectory key for logs, manifests, and packages. Artifact
filenames contain the `target_family` (e.g., `gfx94X`); see
[#3381](https://github.com/ROCm/TheRock/issues/3381) for ongoing work to
propagate artifact group naming consistently.

#### Single-stage CI layout

```
{prefix}/
    {artifact_name}_{component}_{target_family}.tar.xz
    {artifact_name}_{component}_{target_family}.tar.xz.sha256sum
    index-{artifact_group}.html

    logs/{artifact_group}/
        {subproject}_build.log
        {subproject}_configure.log
        {subproject}_install.log
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

#### Multi-arch CI layout

Logs are organized by stage, with a subdirectory per GPU family for per-arch
stages. Generic stages (foundation, compiler-runtime) have no family
subdirectory. Per-arch stages (e.g., math-libs) fan out across GPU
families in parallel, producing identically-named log files (e.g.,
`rocBLAS_build.log`) that are kept separate by the family subdirectory.

```
{prefix}/
    {artifact_name}_{component}_{target_family}.tar.zst
    {artifact_name}_{component}_{target_family}.tar.zst.sha256sum

    logs/{stage_name}/                          (generic stages)
        {subproject}_build.log
        {subproject}_configure.log
        {subproject}_install.log
        ninja_logs.tar.gz

    logs/{stage_name}/{amdgpu_family}/          (per-arch stages)
        {subproject}_build.log
        {subproject}_configure.log
        {subproject}_install.log
        ninja_logs.tar.gz

    python/
        *.whl                                   (generic wheels, e.g., rocm_sdk_core)
        {amdgpu_family}/
            *.whl                               (per-family wheels, e.g., rocm_sdk_devel)
            *.tar.gz                            (sdist)
            index.html
```

Example for a run with foundation + math-libs stages:

```
12345-linux/
    logs/foundation/
        rocm-cmake_build.log
        rocm-cmake_configure.log
        rocm-cmake_install.log
        ninja_logs.tar.gz

    logs/compiler-runtime/
        amd-llvm_build.log
        amd-llvm_configure.log
        amd-llvm_install.log
        ninja_logs.tar.gz

    logs/math-libs/gfx1151/
        rocBLAS_build.log
        MIOpen_build.log
        ninja_logs.tar.gz

    logs/math-libs/gfx110X-all/
        rocBLAS_build.log
        MIOpen_build.log
        ninja_logs.tar.gz
```

Artifacts in multi-arch CI use `.tar.zst` compression (vs `.tar.xz` in
single-stage CI) and are managed by `artifact_manager.py`, not
`post_build_upload.py`. Index pages for logs are generated server-side.

### Bucket selection

The bucket is determined by `_retrieve_bucket_info()` in `workflow_outputs.py`.
See [S3 Buckets](s3_buckets.md) for the full list of buckets and authentication
details.

```
RELEASE_TYPE set? ──Yes──> therock-{RELEASE_TYPE}-artifacts
       │
       No
       │
ROCm/TheRock (not fork)? ──Yes──> therock-ci-artifacts
       │
       No
       │
       └──> therock-ci-artifacts-external
```

Valid `RELEASE_TYPE` values are `dev`, `nightly`, and `prerelease`.

## Python API

### StorageLocation

A frozen dataclass representing a single file or directory in S3 (or a local
staging directory). Backend-agnostic — usable for CI run outputs, release
artifacts, or any S3 path.

```python
from _therock_utils.storage_location import StorageLocation

loc = StorageLocation(
    bucket="therock-ci-artifacts", relative_path="12345-linux/file.tar.xz"
)
loc.s3_uri  # "s3://therock-ci-artifacts/12345-linux/file.tar.xz"
loc.https_url  # "https://therock-ci-artifacts.s3.amazonaws.com/12345-linux/file.tar.xz"
loc.local_path(Path("/tmp/staging"))  # Path("/tmp/staging/12345-linux/file.tar.xz")
```

### WorkflowOutputRoot

A frozen dataclass that computes `StorageLocation` for every output type.

```python
from _therock_utils.workflow_outputs import WorkflowOutputRoot

# Inside a CI workflow (env vars provide bucket info, no API call)
root = WorkflowOutputRoot.from_workflow_run(run_id="12345", platform="linux")

# Fetching artifacts from another run (API call for fork/cutover detection)
root = WorkflowOutputRoot.from_workflow_run(
    run_id="12345", platform="linux", lookup_workflow_run=True
)

# For local development (no API calls, no env vars needed)
root = WorkflowOutputRoot.for_local(run_id="local", platform="linux")

# Location methods — each returns an StorageLocation
root.root()
root.artifact(filename="blas_lib_gfx94X.tar.xz")
root.artifact_index()
root.log_dir(artifact_group="gfx94X-dcgpu")
root.log_stage_dir(stage_name="math-libs", amdgpu_family="gfx1151")
root.log_stage_dir(stage_name="foundation")  # generic stage, no family
root.log_file(artifact_group="gfx94X-dcgpu", filename="build.log")
root.log_index(artifact_group="gfx94X-dcgpu")
root.build_observability(artifact_group="gfx94X-dcgpu")
root.manifest_dir(artifact_group="gfx94X-dcgpu")
root.manifest(artifact_group="gfx94X-dcgpu")
root.python_packages(artifact_group="gfx110X-all")
```

The `lookup_workflow_run` parameter controls whether `from_workflow_run()` calls
the GitHub API to fetch workflow run metadata (for fork detection and bucket
cutover dating). Most callers running inside their own CI workflow do not need
this — environment variables (`GITHUB_REPOSITORY`, `IS_PR_FROM_FORK`) suffice.
Set `lookup_workflow_run=True` when looking up another repository's workflow
run, e.g. when fetching artifacts.

### StorageBackend

An abstract base class for uploading files to S3 or a local directory.
Use `create_storage_backend()` to get the right implementation.

```python
from _therock_utils.storage_backend import create_storage_backend

backend = create_storage_backend()  # S3 (default)
backend = create_storage_backend(staging_dir=Path("/tmp/out"))  # local directory
backend = create_storage_backend(dry_run=True)  # print only

backend.upload_file(source_path, dest_location)
backend.upload_directory(source_dir, dest_location, include=["*.tar.xz*"])
```

Content-type is inferred from file extension — callers don't need to specify it.

### Adding new output types

To add a new output type:

1. Add a method to `WorkflowOutputRoot` that returns `StorageLocation`
1. Add tests to [`build_tools/tests/workflow_outputs_test.py`](/build_tools/tests/workflow_outputs_test.py)
1. Update this document

## Consumers

### Upload scripts

| File                                                                                       | Uses                                                                      |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| [`post_build_upload.py`](/build_tools/github_actions/post_build_upload.py)                 | `WorkflowOutputRoot` + `StorageBackend` for artifacts, logs, manifests    |
| [`post_stage_upload.py`](/build_tools/github_actions/post_stage_upload.py)                 | `WorkflowOutputRoot` + `StorageBackend` for multi-arch stage logs         |
| [`upload_python_packages.py`](/build_tools/github_actions/upload_python_packages.py)       | `WorkflowOutputRoot` + `StorageBackend` for Python wheels and index       |
| [`upload_pytorch_manifest.py`](/build_tools/github_actions/upload_pytorch_manifest.py)     | `WorkflowOutputRoot` + `StorageBackend` for PyTorch manifests             |
| [`upload_test_report_script.py`](/build_tools/github_actions/upload_test_report_script.py) | `WorkflowOutputRoot` for S3 base URI (upload not yet migrated to backend) |

### Download scripts

| File                                                                        | Uses                                                                           |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| [`fetch_artifacts.py`](/build_tools/fetch_artifacts.py)                     | `WorkflowOutputRoot.from_workflow_run(lookup_workflow_run=True)` + `S3Backend` |
| [`find_artifacts_for_commit.py`](/build_tools/find_artifacts_for_commit.py) | `WorkflowOutputRoot.from_workflow_run(workflow_run=...)` for bucket/prefix     |
| [`artifact_backend.py`](/build_tools/_therock_utils/artifact_backend.py)    | `WorkflowOutputRoot` for `S3Backend` construction                              |
| [`artifact_manager.py`](/build_tools/artifact_manager.py)                   | Via `create_backend_from_env()`                                                |
