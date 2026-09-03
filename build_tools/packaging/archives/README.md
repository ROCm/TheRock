# ROCm SDK archive packaging

This directory contains tools for creating ROCm SDK archives - file-tree
representations of an SDK installation that can be extracted and used directly.

Each archive contains a flattened set of ROCm build
[artifacts](/docs/development/artifacts.md) from the ROCm subprojects included
in the build:

```bash
install/      # Arbitrary file path the archive is extracted to
  .kpack/
  bin/
  clients/
  include/
  lib/
  libexec/
  share/
```

Archives are _just_ these raw files. They do not come with "install" steps
such as setting environment variables.

## Archive formats

Currently both Linux and Windows archives use the `.tar.gz` format ("tarballs").
See [PR#4438](https://github.com/ROCm/TheRock/pull/4438) for proposed changes to
archive layout, naming, portability, and integrity requirements.

## Generating archives

### Using the `build_tarballs.py` script

[`build_tarballs.py`](./build_tarballs.py) fetches ROCm build artifacts from a
GitHub Actions workflow run, flattens them into an install-prefix-like layout,
and creates `.tar.gz` archives. It can create per-family archives and, for
builds using split kernel-pack artifacts, a combined multi-architecture archive.

From the repository root, run. This example uses artifacts from
[rockrel release run 33697653391](https://github.com/ROCm/rockrel/actions/runs/33697653391):

```bash
RUN_ID=33697653391
RUN_GITHUB_REPO=ROCm/rockrel
PACKAGE_VERSION=10.1.0a20260903
DIST_AMDGPU_FAMILIES="gfx94X-dcgpu;gfx110X-all;gfx1151;gfx120X-all;gfx90a;gfx950-dcgpu;gfx900;gfx90c;gfx906;gfx908;gfx101X-dgpu;gfx103X-all;gfx1150;gfx1152;gfx1153;gfx125X-dcgpu"
OUTPUT_DIR=build/tarballs

python build_tools/packaging/archives/build_tarballs.py \
  --run-id="$RUN_ID" \
  --run-github-repo="$RUN_GITHUB_REPO" \
  --dist-amdgpu-families="$DIST_AMDGPU_FAMILIES" \
  --platform=linux \
  --package-version="$PACKAGE_VERSION" \
  --output-dir="$OUTPUT_DIR"
```

### Creating an archive directly from workflow run artifacts

For manual packaging, `artifact_manager.py fetch --flatten` can download and
merge the artifact slices from a CI/CD run into one SDK-root directory. That
directory can then be compressed directly:

```bash
RUN_ID=33697653391
RUN_GITHUB_REPO=ROCm/rockrel
ARCHIVE_ROOT=build/archive-root
ARCHIVE_PATH=build/rocm-sdk-ci-linux.tar.gz

python build_tools/artifact_manager.py fetch \
  --run-id="$RUN_ID" \
  --run-github-repo="$RUN_GITHUB_REPO" \
  --stage=all \
  --amdgpu-families=gfx110X-all \
  --expand-family-to-targets \
  --platform=linux \
  --exclude-components=test \
  --exclude-artifacts=fftw3 \
  --output-dir="$ARCHIVE_ROOT" \
  --flatten

tar -czf "$ARCHIVE_PATH" -C "$ARCHIVE_ROOT" .
```

Remove the two `--exclude-*` options to include test and FFTW artifacts. The
output archive contains the flattened SDK tree at its root, without a wrapping
`archive-root/` directory.

### Creating an archive directly from a local build

For a configured local build, the `therock-dist` target directly populates
`build/dist/rocm/`, which has the SDK-root layout expected in an archive:

```bash
cmake --build build --target therock-dist

tar -czf build/rocm-sdk-local-linux.tar.gz -C build/dist/rocm .
```

To exercise artifact creation and flattening as part of the test, build
`therock-artifacts` instead. This creates the separate artifact slices under
`build/artifacts/` and also populates their flattened distribution under
`build/dist/rocm/`:

```bash
cmake --build build --target therock-artifacts

tar -czf build/rocm-sdk-local-linux.tar.gz -C build/dist/rocm .
```

The archive contents reflect the projects, GPU targets, and optional components
enabled in the local CMake configuration. Do not archive `build/artifacts/`
itself: that directory contains separate build-tree-shaped artifact slices, not
the flattened SDK layout.

## Using archives

Current release usage is documented in
[RELEASES.md](../../../RELEASES.md#installing-multi-arch-tarballs).
