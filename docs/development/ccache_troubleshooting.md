# CCache Troubleshooting

This page covers how to diagnose and fix ccache hit rate issues in TheRock CI.

## Introduction

[ccache](https://ccache.dev/) is a compiler cache that speeds up
recompilation by caching previous compilation results and detecting when
the same compilation is being done again. TheRock uses ccache for both
local developer builds and CI builds, with a shared remote cache server
so that cache entries can be reused across CI runners.

[sccache](https://github.com/mozilla/sccache) is an alternative compiler
cache with similar goals. We are evaluating both ccache and sccache for
use across ROCm and downstream frameworks like PyTorch (see
`build_tools/setup_sccache_rocm.py`). The goal is to standardize on a
cache setup that works well across our full ecosystem of projects using
shared infrastructure (e.g., cache servers). This page focuses on ccache,
which is the current default for TheRock CI.

### Key files

| File                                                                                             | Description                                     |
| ------------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| [`build_tools/setup_ccache.py`](../../build_tools/setup_ccache.py)                               | Generates ccache config for local and CI builds |
| [`build_tools/posix_ccache_compiler_check.py`](../../build_tools/posix_ccache_compiler_check.py) | Custom compiler fingerprinting for POSIX        |
| [`build_tools/hack/ccache/`](../../build_tools/hack/ccache/)                                     | Sanity tests and analysis scripts               |

## CI infrastructure

### Remote cache servers

CI uses [bazel-remote](https://github.com/buchgr/bazel-remote) as a shared
remote cache, accessed via ccache's `remote_storage` option with
`layout=bazel`:

| Preset               | Server                                                      | Used by               |
| -------------------- | ----------------------------------------------------------- | --------------------- |
| `github-oss-dev`     | `bazelremote-svc.bazelremote-ns.svc.cluster.local:8080`     | PR builds, postsubmit |
| `github-oss-release` | `bazelremote-svc-rel.bazelremote-ns.svc.cluster.local:8080` | Release builds        |

Both servers are on the Kubernetes cluster, accessible without
authentication from any pod in the cluster.

### In-cluster Redis backend

The `github-oss-redis` preset points ccache's `remote_storage` at an
in-cluster Redis (Valkey) service in the prod build cluster
(`therock-runners-prod`):

| Preset             | Server                                                    | Used by              |
| ------------------ | --------------------------------------------------------- | -------------------- |
| `github-oss-redis` | `redis-ccache-svc.redis-ccache-ns.svc.cluster.local:6379` | Linux CI (default)   |

Like bazel-remote, it is a `ClusterIP` service with **no authentication**,
relying on cluster network isolation. The endpoint can be overridden with the
`CCACHE_REDIS_ENDPOINT` environment variable. It is deployed from
`ROCm/TheRock-Infra` (`helm/redis-ccache`, `deploy-redis-ccache-prod.yaml`).

This is the default remote backend for Linux multi-arch CI builds. Since the
service is in-cluster only, builds must run on the AWS in-cluster pool
(`aws-linux-scale-rocm-prod`). Windows builds are unaffected and continue
using their existing cache setup.

Caveat: ccache's built-in Redis backend has no TLS and is **deprecated for
removal in ccache 4.14/4.15**. This backend works with the pinned 4.11.2 but
has a shelf-life; revisit if/when ccache is upgraded.

### Namespace version

`CCACHE_NAMESPACE_VERSION` in `setup_ccache.py` controls the cache
namespace. Bump it when making hash-affecting config changes (sloppiness,
compiler_check, etc.) to isolate from stale entries. All entries in the
old namespace become unreachable — the cache starts cold, so the first
CI run after a bump will be all misses.

Currently a single namespace is shared across all repos, platforms, and
build configurations. ccache's own hashing (compiler binary, flags,
source content) partitions entries naturally, but if cross-repo or
cross-platform pollution becomes an issue, per-repo or per-platform
namespaces could be introduced.

### Platform differences

|                  | Linux                                                                                                        | Windows                                                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| ccache version   | Pinned in [manylinux Dockerfile](../../dockerfiles/build_manylinux_x86_64.Dockerfile) (4.11.2 as of 2026-05) | Installed via chocolatey in the [Windows workflow](../../.github/workflows/multi_arch_build_windows_artifacts.yml) (4.13.6 as of 2026-05) |
| `compiler_check` | Custom script (hashes binary + shared libs via sha256sum)                                                    | `content` (hashes binary content)                                                                                                         |
| Build container  | Same manylinux Docker image for all jobs                                                                     | Ephemeral VMs with unique workspace GUIDs                                                                                                 |
| Workspace path   | `/__w/TheRock/TheRock/` (consistent)                                                                         | `C:\home\runner\_work\TheRock\TheRock\` (consistent)                                                                                      |
| Build directory  | `/__w/TheRock/TheRock/build/`                                                                                | `B:\build` (volume mount)                                                                                                                 |

### Windows drive mounts

Windows CI runners use Azure VMs where `B:\` is a temporary disk mounted
as a volume mount point to `C:\{GUID}\`. The GUID is unique per VM, so
`B:\build` resolves to different absolute paths on different runners.

The `-resource-dir` flag is explicitly passed to clang in the toolchain
setup (`cmake/therock_subproject.cmake`) to prevent clang from
auto-detecting its resource directory through the resolved mount path.

### Cross-repo cache sharing

Multiple ROCm repos (TheRock, SPIRV-LLVM-Translator, rocm-libraries,
rocm-systems, etc.) use the same `setup_ccache.py` and share the same
bazelremote cache servers. Each repo gets a different workspace path
(e.g., `/_work/{repo}/{repo}` on Linux, `C:\home\runner\_work\{repo}\{repo}`
on Windows). The namespace isolates entries to some extent, but be aware
of cross-repo pollution — entries from one repo may share manifests with
another if they compile the same source files with compatible flags.

## Downloading and inspecting CI logs

### Finding a workflow run

The helper scripts in `build_tools/` can find runs and artifacts
automatically:

```bash
# Find the latest successful run for an artifact group
python build_tools/find_latest_artifacts.py --artifact-group gfx110X-all -v

# Find artifacts for a specific commit
python build_tools/find_artifacts_for_commit.py --commit abc123 \
  --artifact-group gfx110X-all
```

For more manual exploration using the GitHub API directly:

```bash
# Find recent multi-arch CI runs
gh api "repos/ROCm/TheRock/actions/workflows/210763103/runs?per_page=10" \
  -q '.workflow_runs[] | "\(.id) \(.created_at[:16]) \(.conclusion) \(.head_sha[:10])"'

# Find math-libs jobs in a specific run
gh api "repos/ROCm/TheRock/actions/runs/{RUN_ID}/jobs?per_page=100" --paginate \
  -q '.jobs[] | select(.name | contains("math-libs") and contains("gfx1151")) | "\(.id) \(.name) \(.conclusion)"'
```

### Checking ccache stats from a job

The ccache stats are printed in the "Report" step of each build job:

```bash
gh api repos/ROCm/TheRock/actions/jobs/{JOB_ID}/logs 2>/dev/null \
  | grep -A 25 "Cacheable calls"
```

### Downloading ccache logs from S3

Build logs are uploaded to S3 as artifacts. The ccache log archive
contains `ccache.log` (detailed per-compilation log) and
`ccache_stats.log` (per-file statistics).

```
# URL pattern
https://therock-ci-artifacts.s3.amazonaws.com/{RUN_ID}-{PLATFORM}/logs/{STAGE}/{GFX_FAMILY}/ccache_logs.tar.zst

# Example
https://therock-ci-artifacts.s3.amazonaws.com/25465494022-windows/logs/math-libs/gfx1151/ccache_logs.tar.zst
```

To download and extract:

```bash
curl -sL -o ccache_logs.tar.zst \
  "https://therock-ci-artifacts.s3.amazonaws.com/{RUN_ID}-windows/logs/math-libs/gfx1151/ccache_logs.tar.zst"
```

```python
# Extract (requires zstandard package)
import tarfile, io, zstandard

dctx = zstandard.ZstdDecompressor()
with open("ccache_logs.tar.zst", "rb") as f:
    reader = dctx.stream_reader(f)
    with tarfile.open(fileobj=reader, mode="r|") as tf:
        tf.extractall("output_dir")
```

### Using the analysis scripts

```bash
# Analyze a CI run's ccache logs (downloads from S3 automatically)
python build_tools/hack/ccache/analyze_ccache_logs.py \
  --run-id 25465494022 --stage math-libs --gfx gfx1151

# Analyze a local log file
python build_tools/hack/ccache/analyze_ccache_logs.py \
  --log-file /path/to/ccache.log

# Compare hit rates per subproject between two logs (e.g., Linux vs Windows)
python build_tools/hack/ccache/compare_ccache_by_project.py \
  /path/to/linux/ccache.log /path/to/windows/ccache.log
```

### Other log files

The S3 index page lists all available logs for a run:

```
https://therock-ci-artifacts.s3.amazonaws.com/{RUN_ID}-{PLATFORM}/logs/{STAGE}/{GFX_FAMILY}/index.html
```

This includes per-subproject build/configure/install logs, ninja logs,
and the ccache logs archive.

## Symptoms of poor cache hit rates

### Symptom: Low overall hit rate

Check the "Report" step output. A healthy warm-cache run should show
~95%+ hit rates for cacheable calls.

**What to check:**

1. **Is the cache warm?** The first run after a namespace bump or config
   change will be all misses. Check the second run.

1. **Which compiler is missing?** Use `analyze_ccache_logs.py` to break
   down hits/misses by compiler binary. `cl.exe` and `clang++` may have
   different hit rates.

### Symptom: "can't be read" entries in ccache log

```
C:\{GUID}\build\...\__stdarg_va_copy.h is mentioned in a manifest
entry but can't be read (No such file or directory)
```

This means manifest entries reference absolute paths from a different
runner that no longer exist on the current runner.

**Likely cause:** Path instability — the build directory resolves to a
different absolute path on each runner (e.g., GUID-based volume mounts
on Windows).

**Fix:** Ensure all tools see and use stable paths. Passing an explicit
`-resource-dir` to clang (in `therock_subproject.cmake`) can help.

### Symptom: Manifest found but no entries match

ccache finds the manifest (same source + command line + compiler hash)
but all result entries fail verification.

**Likely causes:**

- **Unstable include file paths:** Header paths change between runs
  (see "can't be read" above)
- **Generated headers with volatile content:** Version headers that
  embed git hashes or timestamps change on every build. See
  `THEROCK_FLAG_STAMP_LIBRARY_GIT_VERSIONS` in `FLAGS.cmake` — this
  should be OFF for non-release builds to avoid embedding git hashes
  in `hip_version.h`.
- **Re-staged headers with fresh mtime:** The build system may rewrite
  headers during the stage/install step even when content is unchanged.
  See [issue #5009](https://github.com/ROCm/TheRock/issues/5009).

### Symptom: Different manifest keys across runs for the same file

The direct hash (manifest key) differs between runs even for unchanged
source files.

**Likely causes:**

- **GUID in compiler flags:** Check if `-D` defines or other flags
  contain runner-specific paths (e.g., `-DHIP_COMPILER_FLAGS` embedding
  the resolved `C:\{GUID}\...` path)
- **Different compiler binary:** If the compiler is rebuilt from source
  and `compiler_check = content` is used, verify the binary is
  byte-identical across runs. Use the analysis scripts or compare
  SHA256 hashes of the compiler binary from the CI artifacts.
- **Different ccache version:** Different ccache versions use different
  hash formats. Entries from one version can't be used by another.

## Validating fixes

### Verifying compiler binary reproducibility

Download the compiler binary from two different CI runs and compare:

```bash
# Download from S3 (streams ~600MB, extracts just clang++)
python -c "
import tarfile, io, urllib.request, zstandard, hashlib
url = 'https://therock-ci-artifacts.s3.amazonaws.com/{RUN_ID}-windows/amd-llvm_run_generic.tar.zst'
with urllib.request.urlopen(url) as resp:
    data = resp.read()
dctx = zstandard.ZstdDecompressor()
reader = dctx.stream_reader(io.BytesIO(data))
with tarfile.open(fileobj=reader, mode='r|') as tf:
    for member in tf:
        if 'clang++.exe' in member.name and member.isfile():
            f = tf.extractfile(member)
            print(f'{hashlib.sha256(f.read()).hexdigest()}  {member.name}')
            break
"
```

Both runs should produce the same SHA256. If they don't, check that
`/Brepro` is being applied (see `therock_subproject.cmake`).

### Verifying path stability

Check the ccache log for GUID or runner-specific paths:

```bash
# Count "can't be read" entries (should be 0)
grep -c "can't be read" ccache.log

# Check for GUID patterns in the log
grep -oP "[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}" ccache.log | sort -u

# Check the setup_ccache.py log output for resolved paths
grep "setup_ccache" job_log.txt
```

### Verifying flag stability

Compare command lines for the same source file across two runs using the
ccache logs. Key things to check:

- `-I` and `-isystem` paths should be consistent
- `-D` defines should not contain runner-specific values
- `-resource-dir`, `--hip-path`, and `--hip-device-lib-path` should use stable
  paths
