# CI Run Outputs Layout

This document describes the directory structure for workflow run outputs stored in S3.
For information about how build artifacts are created locally, see [artifacts.md](artifacts.md).

## Overview

A "run output" is anything produced by a CI workflow run:

- **Build artifacts** - `.tar.xz` or `.tar.zst` archives of compiled components
- **Logs** - Build logs, ninja timing logs, log indexes
- **Manifests** - `therock_manifest.json` containing source control metadata
- **Reports** - Build time analysis, test reports
- *(and other files which may be added later)*

Each job for a given workflow run uploads its outputs to a central directory and
there is a 1:1 mapping between a workflow run ID and a run outputs directory.

> [!NOTE]
> As our current CI/CD system is built on GitHub Actions, we use the "workflow run"
> and "workflow run id" terminology from GitHub Actions, but this system could also
> work using a different CI/CD system.

## S3 Structure

### S3 Buckets

Run outputs are stored in public S3 buckets. The bucket used depends on the source repository and release type:

| Bucket                          | Purpose                                    | Browse                                                            |
| ------------------------------- | ------------------------------------------ | ----------------------------------------------------------------- |
| `therock-ci-artifacts`          | Main CI artifacts from ROCm/TheRock        | [Browse](https://therock-ci-artifacts.s3.amazonaws.com/)          |
| `therock-ci-artifacts-external` | CI artifacts from forks and external repos | [Browse](https://therock-ci-artifacts-external.s3.amazonaws.com/) |
| `therock-nightly-artifacts`     | Nightly release builds                     | [Browse](https://therock-nightly-artifacts.s3.amazonaws.com/)     |
| `therock-release-artifacts`     | Official release builds                    | [Browse](https://therock-release-artifacts.s3.amazonaws.com/)     |

Legacy buckets (for older workflow runs):

| Bucket                       | Purpose                        | Browse                                                         |
| ---------------------------- | ------------------------------ | -------------------------------------------------------------- |
| `therock-artifacts`          | Legacy main CI artifacts       | [Browse](https://therock-artifacts.s3.amazonaws.com/)          |
| `therock-artifacts-external` | Legacy external repo artifacts | [Browse](https://therock-artifacts-external.s3.amazonaws.com/) |

### Root Path

Each workflow run has a root directory in S3:

```
s3://{bucket}/{external_repo}{run_id}-{platform}/
```

Where:

| Field           | Description                                         | Example                   |
| --------------- | --------------------------------------------------- | ------------------------- |
| `bucket`        | S3 bucket name                                      | `therock-ci-artifacts`    |
| `external_repo` | Empty for ROCm/TheRock, `{owner}-{repo}/` for forks | (empty), `MyOrg-TheRock/` |
| `run_id`        | GitHub Actions workflow run ID                      | `12345678901`             |
| `platform`      | Build platform                                      | `linux`, `windows`        |

### Directory Layout

```
{run_id}-{platform}/
├── {name}_{component}_{family}.tar.xz        # Build artifacts (at root)
├── {name}_{component}_{family}.tar.xz.sha256sum
├── index-{artifact_group}.html               # Per-group artifact index
├── logs/{artifact_group}/
│   ├── *.log                                 # Build logs
│   ├── ninja_logs.tar.gz                     # Ninja timing logs
│   ├── index.html                            # Log index
│   └── build_time_analysis.html              # Build timing (Linux only)
└── manifests/{artifact_group}/
    └── therock_manifest.json                 # Build manifest
```

### Naming Conventions

| Field            | Description                    | Examples                                  |
| ---------------- | ------------------------------ | ----------------------------------------- |
| `name`           | Artifact name                  | `blas`, `core-hip`, `amd-llvm`            |
| `component`      | Role of files contained        | `dev`, `lib`, `run`, `test`, `dbg`, `doc` |
| `family`         | Target GPU family or `generic` | `generic`, `gfx94X`, `gfx1100`            |
| `artifact_group` | Build variant identifier       | `gfx94X-dcgpu`, `gfx950-dcgpu-asan`       |

## Multi-Platform / Multi-Group Organization

A single workflow run may produce outputs for multiple platforms and artifact groups.
Each platform gets its own root directory:

```
s3://therock-ci-artifacts/
├── 12345678901-linux/
│   ├── *.tar.xz (artifacts for all Linux artifact groups)
│   ├── index-gfx94X-dcgpu.html
│   ├── index-gfx110X-all.html
│   ├── logs/gfx94X-dcgpu/
│   ├── logs/gfx110X-all/
│   └── ...
└── 12345678901-windows/
    ├── *.tar.xz (artifacts for all Windows artifact groups)
    ├── index-gfx110X-dgpu.html
    └── ...
```

Multiple CI jobs (different GPU families, build variants) upload to the same run
directory, differentiated by `artifact_group` in subdirectory names and index
filenames.

## Fork/External Repository Handling

Builds from forks or external repositories use a different S3 bucket and include
a prefix to namespace their outputs:

| Source        | Bucket                          | External Repo Prefix |
| ------------- | ------------------------------- | -------------------- |
| ROCm/TheRock  | `therock-ci-artifacts`          | (empty)              |
| Fork/external | `therock-ci-artifacts-external` | `{owner}-{repo}/`    |

Example paths:

```
# Main repo
s3://therock-ci-artifacts/12345678901-linux/

# Fork (e.g., MyOrg/TheRock)
s3://therock-ci-artifacts-external/MyOrg-TheRock/12345678901-linux/
```

## Public Access

Artifacts are publicly accessible via HTTPS:

```
https://{bucket}.s3.amazonaws.com/{prefix}/...
```

Example URLs:

- Artifact index: https://therock-ci-artifacts-external.s3.amazonaws.com/ROCm-TheRock/21155514542-linux/index-gfx110X-all.html
- Build logs: https://therock-ci-artifacts-external.s3.amazonaws.com/ROCm-TheRock/21155514542-linux/logs/gfx110X-all/index.html
- Manifest: https://therock-ci-artifacts-external.s3.amazonaws.com/ROCm-TheRock/21155514542-linux/manifests/gfx110X-all/therock_manifest.json

## Using RunOutputRoot

All path computation should go through the `RunOutputRoot` class in
`build_tools/_therock_utils/run_outputs.py`. This ensures consistent paths
across all tools.

### Basic Usage

```python
from _therock_utils.run_outputs import RunOutputRoot

root = RunOutputRoot.from_workflow_run(
    github_repository="ROCm/TheRock",  # Or omit to infer from GITHUB_REPOSITORY
    run_id="12345678901",
    platform="linux",
)

# Access paths
print(root.s3_uri)  # s3://bucket/12345678901-linux
print(root.https_url)  # https://bucket.s3.amazonaws.com/...
print(root.artifact_index_url("gfx94X-dcgpu"))  # ...index-gfx94X-dcgpu.html
print(root.logs_s3_uri("gfx94X-dcgpu"))  # s3://bucket/.../logs/gfx94X-dcgpu
print(
    root.manifest_url("gfx94X-dcgpu")
)  # ...manifests/gfx94X-dcgpu/therock_manifest.json
```

### Local Development

For local testing without S3:

```python
root = RunOutputRoot.for_local(run_id="local-test", platform="linux")
local_dir = root.local_path(Path("/tmp/staging"))
# Returns: /tmp/staging/local-test-linux/
```

### Adding New Output Types

To add a new output type (e.g., Python packages, native packages, reports):

1. Add methods to `RunOutputRoot` following the existing patterns:

   - `{type}_prefix(artifact_group)` - S3 key prefix
   - `{type}_s3_uri(artifact_group)` - Full S3 URI
   - `{type}_s3_key(artifact_group, filename)` - S3 key for specific file
   - `{type}_url(artifact_group, filename)` - Public HTTPS URL (if applicable)

1. Update the upload script to use the new methods

1. Update this documentation

## Related Files

| File                                                                                                        | Purpose                                       |
| ----------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| [`build_tools/_therock_utils/run_outputs.py`](/build_tools/_therock_utils/run_outputs.py)                   | `RunOutputRoot` class for path computation    |
| [`build_tools/_therock_utils/artifact_backend.py`](/build_tools/_therock_utils/artifact_backend.py)         | Storage backend abstraction (S3 or local)     |
| [`build_tools/github_actions/post_build_upload.py`](/build_tools/github_actions/post_build_upload.py)       | Uploads artifacts, logs, manifests to S3      |
| [`build_tools/github_actions/github_actions_utils.py`](/build_tools/github_actions/github_actions_utils.py) | GitHub API utilities (workflow queries, etc.) |
