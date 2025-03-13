# Releases

This is a quick overview of how to consume our current build and release [artifacts](docs/development/artifacts.md).

## Current state

Currently, we produce build artifacts as part of our CI workflows ([example](.github/workflows/build_linux_packages.yml)) as well as part of our release. As the project is still not ready for production (see [ROADMAP](ROADMAP.md)) this doc assumes you are already familiar with how to use ROCm. If not - you should not start here, please start at [ROCm](https://github.com/ROCm/ROCm).

## Installing from Tarballs

The quickest way to "install" our artifacts is to leverage the `build_tools/fileset_tool.py artifact-flatten` command. You will need to have a [checkout](README.md#Checkout-Sources) to leverage this tool and a Python environment.

Here's a quick example assuming you copied the all the tar files into `${BUILD_ARTIFACTS_DIR}` to "install" TheRock into `${BUILD_ARTIFACTS_DIR}/output_dir`

```bash
echo "Unpacking artifacts"
pushd "${BUILD_ARTIFACTS_DIR}"
mkdir output_dir
python build_tools/fileset_tool.py artifact-flatten *.tar.xz -o output_dir --verbose
popd
```

## Testing your installation

The quickest way is to run `rocminfo`

```bash
echo "Running rocminfo"
pushd "${BUILD_ARTIFACTS_DIR}"
./output_dir/bin/rocminfo
popd
```
## Where to Get Artifacts

- [Releases](releases): Our releases page has the latest "developer" release of our tarball artifacts and source code.
- [Packages](https://github.com/orgs/ROCm/packages?repo_name=TheRock): We currently publish docker images for LLVM targets we support (as well as a container for our build machines)
- [CI](https://github.com/ROCm/TheRock/actions/workflows/ci.yml?query=branch%3Amain+is%3Asuccess): Each of our latest green builds has its own artifacts you can leverage. This is the latest and greatest! We will eventually support a nightly release that is at a higher quality bar than CI.
