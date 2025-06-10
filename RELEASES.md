# Releases

This page describes how to install and use our release artifacts for ROCm and
external builds like PyTorch.

These other pages provide useful context:

- [Roadmap for support](ROADMAP.md)
- [Build artifacts overview](docs/development/artifacts.md)

## Disclaimers and current status

Currently, we produce build artifacts as part of our Continuous Integration (CI)
build/test workflows as well as part of Continuous Delivery (CD) nightly
releases.

> [!WARNING]
> These instructions assume familiarity with how to use ROCm. Please see
> https://rocm.docs.amd.com/ for general information about the ROCm software
> platform.

## Installing using pip

We recommend installing ROCm and projects like PyTorch via `pip`, the
[Python package installer](https://packaging.python.org/en/latest/guides/tool-recommendations/).

### Support status

|         | ROCm Python packages | PyTorch Python packages                                             |
| ------- | -------------------- | ------------------------------------------------------------------- |
| Linux   | ✅ Supported         | 🟡 In progress ([#703](https://github.com/ROCm/TheRock/issues/703)) |
| Windows | ⚪ Planned           | ⚪ Planned                                                          |

### Installing ROCm Python packages

> [!TIP]
> We highly recommend working within a [Python virtual environment](https://docs.python.org/3/library/venv.html)
>
> ```bash
> python -m venv .venv
> source .venv/bin/activate
> ```
>
> Multiple virtual environments can be present on a system at a time, allowing you to switch between them at will.

> [!WARNING]
> If you _really_ want a system-wide install, you can pass `--break-system-packages` to `pip` outside a virtual enivornment.
> In this case, commandline interface shims for executables are installed to `/usr/local/bin`, which normally has precedence over `/usr/bin` and might therefore conflict with a previous installation of ROCm.

To install TheRock from pip, you must provide a link to the index page for your
desired GPU architecture. One of the following commands can be used to install
`rocm[libraries,devel]`:

<!-- TODO: mapping from product name to gfx family -->

<!-- TODO: link to known issues -->

#### gfx94X-dcgpu

```bash
python -m pip install \
  --pre --find-links https://therock-nightly-python.s3.us-east-2.amazonaws.com/gfx94X-dcgpu/index.html \
  rocm[libraries,devel]
```

#### gfx110X-dgpu

```bash
python -m pip install \
  --pre --find-links https://therock-nightly-python.s3.us-east-2.amazonaws.com/gfx110X-dgpu/index.html \
  rocm[libraries,devel]
```

#### gfx1151

```bash
python -m pip install \
  --pre --find-links https://therock-nightly-python.s3.us-east-2.amazonaws.com/gfx1151/index.html \
  rocm[libraries,devel]
```

#### gfx120X-all

```bash
python -m pip install \
  --pre --find-links https://therock-nightly-python.s3.us-east-2.amazonaws.com/gfx120X-all/index.html \
  rocm[libraries,devel]
```

### Installing PyTorch

Coming soon!

<!-- TODO: add `torch` to install commands
       * needs new build to be compatible with 'rocm' instead of 'rocm-sdk'
       * For 'rocm-sdk', need an environment workaround -->

## Using our tarballs

Here's a quick way assuming you copied the all the tar files into `${BUILD_ARTIFACTS_DIR}` to "install" TheRock into `${BUILD_ARTIFACTS_DIR}/output_dir`

### From release builds

Our releases are already flattened and simply need untarring, follow the below instructions.

```bash
echo "Unpacking artifacts"
pushd "${BUILD_ARTIFACTS_DIR}"
mkdir output_dir
tar -xf *.tar.gz -C output_dir
popd
```

### From per-commit CI builds

Our CI builds artifacts which need to be "flattened" by the `build_tools/fileset_tool.py artifact-flatten` command before they can be used. You will need to have a checkout (see for example [Clone and fetch sources](https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md#clone-and-fetch-sources)) in `${SOURCE_DIR}` to use this tool and a Python environment.

```bash
echo "Unpacking artifacts"
pushd "${BUILD_ARTIFACTS_DIR}"
mkdir output_dir
python "${SOURCE_DIR}/build_tools/fileset_tool.py artifact-flatten *.tar.xz -o output_dir --verbose
popd
```

### From `install_rocm_from_artifacts.py`

This script installs ROCm community builds produced by TheRock from either a developer/nightly tarball, a specific CI runner build or an already existing installation of TheRock. This script is used by CI and can be used locally.

Examples:

- Downloads all gfx94X S3 artifacts from [GitHub CI workflow run 15052158890](https://github.com/ROCm/TheRock/actions/runs/15052158890) to the default output directory `therock-build`:

  ```
  python build_tools/install_rocm_from_artifacts.py --run-id 15052158890 --amdgpu-family gfx94X-dcgpu --tests
  ```

- Downloads the version `6.4.0rc20250516` gfx110X artifacts from GitHub release tag `nightly-tarball` to the specified output directory `build`:

  ```
  python build_tools/install_rocm_from_artifacts.py --release 6.4.0rc20250516 --amdgpu-family gfx110X-dgpu --output-dir build
  ```

- Downloads the version `6.4.0.dev0+e015c807437eaf32dac6c14a9c4f752770c51b14` gfx110X artifacts from GitHub release tag `dev-tarball` to the default output directory `therock-build`:

  ```
  python build_tools/install_rocm_from_artifacts.py --release 6.4.0.dev0+e015c807437eaf32dac6c14a9c4f752770c51b14 --amdgpu-family gfx110X-dgpu
  ```

Select your AMD GPU family from this file [therock_amdgpu_targets.cmake](https://github.com/ROCm/TheRock/blob/59c324a759e8ccdfe5a56e0ebe72a13ffbc04c1f/cmake/therock_amdgpu_targets.cmake#L44-L81)

By default for CI workflow retrieval, all artifacts (excluding test artifacts) will be downloaded. For specific artifacts, pass in the flag such as `--rand` (RAND artifacts) For test artifacts, pass in the flag `--tests` (test artifacts). For base artifacts only, pass in the flag `--base-only`

## Testing your installation

The quickest way is to run `rocminfo`

```bash
echo "Running rocminfo"
pushd "${BUILD_ARTIFACTS_DIR}"
./output_dir/bin/rocminfo
popd
```

## Where to get artifacts

- [Releases](https://github.com/ROCm/TheRock/releases): Our releases page has the latest "developer" release of our tarball artifacts and source code.

- [Packages](https://github.com/orgs/ROCm/packages?repo_name=TheRock): We currently publish docker images for LLVM targets we support (as well as a container for our build machines)

- [Per-commit CI builds](https://github.com/ROCm/TheRock/actions/workflows/ci.yml?query=branch%3Amain+is%3Asuccess): Each of our latest passing CI builds has its own artifacts you can leverage. This is the latest and greatest! We will eventually support a nightly release that is at a higher quality bar than CI. Note a quick recipe for getting all of these from the s3 bucket is to use this quick command `aws s3 cp s3://therock-artifacts . --recursive --exclude "*" --include "${RUN_ID}-${OPERATING_SYSTEM}/*.tar.xz" --no-sign-request` where ${RUN_ID} is the runner id you selected (see the URL). Check the [AWS docs](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) to get the aws cli.
