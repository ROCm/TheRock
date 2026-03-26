# CI Workflow Outputs Layout

This document describes the directory layout for outputs produced by TheRock's
CI workflow runs, and the Python modules for computing paths, uploading, and
downloading those outputs.

## Overview

Every CI workflow run produces a set of outputs (build artifacts, logs,
manifests, python packages) that are uploaded to S3. Three modules in
`_therock_utils` handle the path computation and I/O:

| Module             | Role                         | Key types                                                              |
| ------------------ | ---------------------------- | ---------------------------------------------------------------------- |
| `storage_location` | Backend-agnostic location    | `StorageLocation`                                                      |
| `workflow_outputs` | CI path computation (no I/O) | `WorkflowOutputRoot`                                                   |
| `storage_backend`  | Upload I/O (write)           | `StorageBackend`, `S3StorageBackend`, `LocalStorageBackend`            |
| `artifact_backend` | Download I/O (read)          | `ArtifactBackend`, `S3Backend`, `LocalDirectoryBackend`, `HTTPBackend` |

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

`artifact_group` is the CI matrix variant, composed of a target family plus an
optional variant suffix (e.g., `gfx94X-dcgpu`, `gfx94X-dcgpu-asan`). It is
used as the subdirectory key for logs, manifests, and packages. Artifact
filenames contain the `target_family` (e.g., `gfx94X`); see
[#3381](https://github.com/ROCm/TheRock/issues/3381) for ongoing work to
propagate artifact group naming consistently.

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

Legacy (pre-cutover):
  Runs before 2025-11-11 (TheRock #2046) use the old bucket names:
    therock-ci-artifacts          → therock-artifacts
    therock-ci-artifacts-external → therock-artifacts-external
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

#### Custom URL Schemas

StorageLocation supports custom URL schemas for both S3 URIs and HTTPS URLs.
Schemas use Python string formatting with `{bucket}` and `{path}` placeholders:

```python
# Custom S3 schema
loc = StorageLocation(
    bucket="my-bucket",
    relative_path="12345-linux/file.tar.xz",
    s3_url_schema="custom-s3://{bucket}/prefix/{path}",
)
loc.s3_uri  # "custom-s3://my-bucket/prefix/12345-linux/file.tar.xz"

# Custom HTTPS schema
loc = StorageLocation(
    bucket="my-bucket",
    relative_path="12345-linux/file.tar.xz",
    https_url_schema="https://cdn.example.com/{bucket}/{path}",
)
loc.https_url  # "https://cdn.example.com/my-bucket/12345-linux/file.tar.xz"
```

When `s3_url_schema` or `https_url_schema` are `None` (the default), the following
defaults are used:

- S3 URI: `s3://{bucket}/{path}`
- HTTPS URL: `https://{bucket}.s3.amazonaws.com/{path}`

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

# Location methods — each returns a StorageLocation
root.root()
root.artifact(filename="blas_lib_gfx94X.tar.xz")
root.artifact_index(artifact_group="gfx94X-dcgpu")
root.log_dir(artifact_group="gfx94X-dcgpu")
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

#### Custom URL Schemas

WorkflowOutputRoot accepts optional schema parameters that are propagated to all
`StorageLocation` instances it creates:

```python
# Custom HTTPS schema (e.g., for CDN)
root = WorkflowOutputRoot.from_workflow_run(
    run_id="12345",
    platform="linux",
    https_url_schema="https://cdn.example.com/{bucket}/{path}",
)
root.artifact("test.tar.xz").https_url
# → "https://cdn.example.com/therock-ci-artifacts/12345-linux/test.tar.xz"

# Custom S3 schema (e.g., for S3-compatible storage)
root = WorkflowOutputRoot.from_workflow_run(
    run_id="12345", platform="linux", s3_url_schema="s3-custom://{bucket}/prefix/{path}"
)
root.artifact("test.tar.xz").s3_uri
# → "s3-custom://therock-ci-artifacts/prefix/12345-linux/test.tar.xz"

# Custom bucket naming (e.g., for different environments)
root = WorkflowOutputRoot.from_workflow_run(
    run_id="12345", platform="linux", bucket_schema="mycompany-{release_type}-builds"
)
# When RELEASE_TYPE=dev, bucket will be "mycompany-dev-builds"
```

All three schemas can be passed to `artifact_manager.py` and `post_build_upload.py`
via command line arguments:

```bash
# Using custom HTTPS URL for CDN
python build_tools/artifact_manager.py fetch \
    --https-url-schema "https://cdn.example.com/{bucket}/{path}" \
    --stage math-libs

# Using custom S3 and bucket schemas
python build_tools/github_actions/post_build_upload.py \
    --s3-url-schema "s3-custom://{bucket}/data/{path}" \
    --bucket-schema "mycompany-{release_type}-artifacts" \
    --artifact-group gfx94X-dcgpu
```

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

### ArtifactBackend

An abstract base class for downloading artifacts from S3, local directories, or HTTP servers.
Use `create_backend_from_env()` to get the right implementation based on environment variables.

```python
from _therock_utils.artifact_backend import create_backend_from_env

# With S3 credentials → S3Backend
backend = create_backend_from_env(gfx_families=["gfx94X-dcgpu"])

# With THEROCK_LOCAL_STAGING_DIR → LocalDirectoryBackend
os.environ["THEROCK_LOCAL_STAGING_DIR"] = "/tmp/staging"
backend = create_backend_from_env(gfx_families=["gfx94X-dcgpu"])

# With THEROCK_AMDGPU_FAMILIES (no S3 credentials) → HTTPBackend (read-only)
os.environ["THEROCK_AMDGPU_FAMILIES"] = "gfx94X-dcgpu,gfx1200"
backend = create_backend_from_env()

# Download artifact
backend.download_artifact("blas_lib_gfx94X.tar.zst", Path("/tmp/blas.tar.zst"))
```

The HTTPBackend downloads artifacts via public HTTPS URLs using `StorageLocation.https_url`:

- Example: `https://therock-ci-artifacts.s3.amazonaws.com/12345-linux/blas_lib_gfx94X.tar.zst`
- Fork prefixes, bucket selection, and all path logic handled automatically via `WorkflowOutputRoot`

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
| [`upload_python_packages.py`](/build_tools/github_actions/upload_python_packages.py)       | `WorkflowOutputRoot` + `StorageBackend` for Python wheels and index       |
| [`upload_pytorch_manifest.py`](/build_tools/github_actions/upload_pytorch_manifest.py)     | `WorkflowOutputRoot` + `StorageBackend` for PyTorch manifests             |
| [`upload_test_report_script.py`](/build_tools/github_actions/upload_test_report_script.py) | `WorkflowOutputRoot` for S3 base URI (upload not yet migrated to backend) |

### Download scripts

| File                                                                        | Uses                                                                           |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| [`fetch_artifacts.py`](/build_tools/fetch_artifacts.py)                     | `WorkflowOutputRoot.from_workflow_run(lookup_workflow_run=True)` + `S3Backend` |
| [`find_artifacts_for_commit.py`](/build_tools/find_artifacts_for_commit.py) | `WorkflowOutputRoot.from_workflow_run(workflow_run=...)` for bucket/prefix     |
| [`artifact_backend.py`](/build_tools/_therock_utils/artifact_backend.py)    | `WorkflowOutputRoot` for `S3Backend`/`HTTPBackend` construction                |
| [`artifact_manager.py`](/build_tools/artifact_manager.py)                   | Via `create_backend_from_env()` (supports S3, Local, and HTTP backends)        |
