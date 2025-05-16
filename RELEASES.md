# Releases

This is a quick overview of how to consume our current build and release [artifacts](docs/development/artifacts.md).

## Current state

Currently, we produce build artifacts as part of our CI workflows ([example](.github/workflows/build_linux_packages.yml)) as well as part of our release. As the project is still not ready for production (see [ROADMAP](ROADMAP.md)) this doc assumes you are already familiar with how to use ROCm. If not - you should not start here, please start at [ROCm](https://github.com/ROCm/ROCm).

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

Our CI builds artifacts need to be flattened to be used. Leverage the `build_tools/fileset_tool.py artifact-flatten` command. You will need to have a [checkout](README.md#Checkout-Sources) in ${SOURCE_DIR} to leverage this tool and a Python environment.

```bash
echo "Unpacking artifacts"
pushd "${BUILD_ARTIFACTS_DIR}"
mkdir output_dir
python "${SOURCE_DIR}/build_tools/fileset_tool.py artifact-flatten *.tar.xz -o output_dir --verbose
popd
```

### From `install_rocm_from_artifacts.py`

This script installs ROCm community builds produced by TheRock from either a developer/nightly release, a specific CI runner build or an already existing installation of TheRock. This script is used by CI and can be used locally.

Examples:

- Downloads all gfx94X S3 artifacts from [GitHub CI workflow run 14474448215](https://github.com/ROCm/TheRock/actions/runs/14474448215) to the default output directory `therock-build`:

  ```
  python build_tools/install_rocm_from_artifacts.py --run-id 14474448215 --amdgpu-family gfx94X-dcgpu --tests
  ```

- Downloads the version `6.4.0rc20250416` gfx110X artifacts from GitHub release tag `nightly-release` to the specified output directory `build`:

  ```
  python build_tools/install_rocm_from_artifacts.py --release 6.4.0rc20250416 --amdgpu-family gfx110X-dgpu --output-dir build
  ```

- Downloads the version `6.4.0.dev0+8f6cdfc0d95845f4ca5a46de59d58894972a29a9` gfx120X artifacts from GitHub release tag `dev-release` to the default output directory `therock-build`:

  ```
  python build_tools/install_rocm_from_artifacts.py --release 6.4.0.dev0+8f6cdfc0d95845f4ca5a46de59d58894972a29a9 --amdgpu-family gfx120X-all
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
