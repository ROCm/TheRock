# Git Maintenance Chores

This is a running log of various chores that may need to be carried out to
manage the project sources and upstream connections.

## Rebase Sub-Projects (happy path)

If there are no problematic local patches that conflict with upstream projects,
then this command is sufficient to fast-forward all submodules to upstream
heads:

```
./build_tools/fetch_sources.py --remote --no-apply-patches
# Capture new submodule heads, inspect and ensure things look sound.
git commit -a -m "Rebase submodules (for conflict prep)"
```

In the above, it is important to capture the submodule heads in a pristine
state, prior to any patches being applied. We'll squash all of the commits to
land.

Apply patches and validate:

```
./build_tools/fetch_sources.py
```

If that succeeds, create a PR and make sure that the CI passes.
For the above, the script `build_tools/bump_submodules.py` can be used,
which is however still under development.

### Resolving Conflicts

If the above fails, it will most likely be when applying patches and you will
get a message indicating that the `git am` command failed and left some state
in a submodule. When this happens, you should clean up by going into the
submodule, looking around and aborting the command. Then proceed to the
next section for rebasing with conflicts:

```
git am --abort
```

You can return the source tree to a consistent state by running (without
arguments):

```
./build_tools/fetch_sources.py --no-apply-patches
```

You will then want to iterate on any failing `git am` command, most likely
simply deleting patch files that are no longer needed (have been subsumed by
an upstream patch). Once your `patches/` directory is clean for the new heads,
do a final `./build_tools/fetch_sources.py`, add all files in a commit and
create a PR.

If modifying patches for a sub-project, you will need to resave its patch
stream. This can be done with a command like:

```
./build_tools/save_patches.sh origin/master Tensile amd-mainline
```

The first argument is the sub-project ToT commit, prior to any local patches.
The second is the submodule name (not path). The third is the sub-directory
within `patches/` that contains the project patches. The first argument is
project specific, unfortunately and can be found by looking at `git log` for
the base branch/tag (for using a symbolic tag).

The base ref can also be obtained by looking at `git submodule status`, assuming
that a commit has been made for the submodule in a pristine state. Example:

```
 b91661c88a0fd7f5848a60212839bfc2496ff932 math-libs/BLAS/Tensile (v2.2.3-4910-gb91661c8)
```

In this case, the commit `b91661c88a0fd7f5848a60212839bfc2496ff932` should be
the base ref. It is best to look at `git log` to verify, as it is quite
easy to get into an inconsistent state, and it is hard to spot errors in
commit hashes.

## Switching a submodule branch

### For submodules with no patches

Using the HIP submodule as an example:

1. Edit [`.gitmodules`](/.gitmodules) to switch the branch:

   ```diff
   [submodule "HIP"]
       path = core/HIP
       url = https://github.com/ROCm/HIP.git
   -	branch = amd-mainline
   +	branch = amd-staging
   ```

1. Change the submodule commit:

   ```bash
   pushd core/HIP
   git fetch origin amd-staging
   git checkout amd-staging
   git pull
   popd
   ```

1. Commit those changes:

   ```bash
   git add .gitmodules
   git add core/HIP
   # ...
   ```

1. Check that `fetch_sources.py` runs successfully:

   ```bash
   python ./build_tools/fetch_sources.py
   ```

### For submodules with patches

These steps are similar to
[Rebase Sub-Projects](#rebase-sub-projects-happy-path).

Using the CLR submodule as an example:

1. Edit [`.gitmodules`](/.gitmodules) to switch the branch:

   ```diff
   [submodule "clr"]
       path = core/clr
       url = https://github.com/ROCm/clr.git
   -	branch = amd-mainline
   +	branch = amd-staging
   ```

1. Fetch just that project from the remote with no patches:

   ```bash
   python ./build_tools/fetch_sources.py \
     --remote \
     --no-apply-patches \
     --system-projects clr \
     --no-include-math-libs \
     --no-include-ml-frameworks
   ```

1. Commit the new submodule head

   ```bash
   git commit -a -m "Rebase submodules (for conflict prep)"
   ```

1. Apply patches and validate:

   ```bash
   python ./build_tools/fetch_sources.py
   ```

## Updating a third-party mirror

Projects in [`third-party/`](../../third-party/) use sources mirrored to AWS
so we maintain control over all build dependencies and can produce hermetic
builds (requiring no network access) as needed.

These are declared like so:

```cmake
therock_subproject_fetch(therock-msgpack-cxx-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/msgpack/msgpack-c/releases/download/cpp-7.0.0/msgpack-cxx-7.0.0.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/msgpack-cxx-7.0.0.tar.gz
  URL_HASH SHA256=7504b7af7e7b9002ce529d4f941e1b7fb1fb435768780ce7da4abaac79bb156f
)
```

To update one of these dependencies:

1. Find the version you want to update to, e.g. by choosing the latest release
   from a page like https://github.com/msgpack/msgpack-c/releases

1. Download the source archive, either uploaded as an asset to the release (like
   `https://github.com/msgpack/msgpack-c/releases/download/cpp-7.0.0/msgpack-cxx-7.0.0.tar.gz`)
   or generated from the repository at a tagged commit (like
   `https://github.com/Dobiasd/frugally-deep/archive/refs/tags/v0.15.31.tar.gz`).

1. Compute the SHA256 checksum of the file.

   - On Linux, you can run `sha256sum [NAME NAME]`
   - On Windows, you can run `Get-FileHash [FILE NAME]` from powershell

1. Sign in to AWS and upload the file to
   https://us-east-2.console.aws.amazon.com/s3/buckets/rocm-third-party-deps

1. Update the comment, URL, and URL_HASH in the CMakeLists.txt file

1. Test the build and make any necessary changes to CMake project configuration

## Adding a new submodule

Submodules are tracked in `.gitmodules` and are managed by
[`fetch_sources.py`](../../build_tools/fetch_sources.py). See
https://git-scm.com/docs/git-submodule for general instructions.

To add a new submodule, using https://github.com/ROCm/rocRoller as an example:

1. Add the submodule using git, specifying options for the name, branch, and
   path:

   ```bash
   git submodule add \
      --name rocRoller \
      -b main \
      https://github.com/ROCm/rocRoller.git \
      math-libs/BLAS/rocRoller
   ```

   This should add a few lines to [`.gitmodules`](../../.gitmodules). Check that
   the new lines there are similar to the existing lines in the file:

   ```diff
   $ git diff --staged
   diff --git a/.gitmodules b/.gitmodules
   index 9b4d996..e5a5ec9 100644
   --- a/.gitmodules
   +++ b/.gitmodules
   @@ -140,3 +140,6 @@
         path = ml-libs/composable_kernel
         url = https://github.com/ROCm/composable_kernel.git
         branch = develop
   +[submodule "rocRoller"]
   +       path = math-libs/BLAS/rocRoller
   +       url = https://github.com/ROCm/rocRoller.git
   +       branch = main
   ```

1. Edit [`fetch_sources.py`](../../build_tools/fetch_sources.py), adding the
   submodule name to one of the project lists and adding any special handling
   (e.g. disabling on Windows) as needed:

   ```diff
   diff --git a/build_tools/fetch_sources.py b/build_tools/fetch_sources.py
   index 18746f1..ebf534e 100755
   --- a/build_tools/fetch_sources.py
   +++ b/build_tools/fetch_sources.py
   @@ -263,6 +271,7 @@ def main(argv):
               "rocPRIM",
               "rocRAND",
   +           "rocRoller",
               "rocSOLVER",
               "rocSPARSE",
               "rocThrust",
   ```

1. Run [`fetch_sources.py`](../../build_tools/fetch_sources.py) to initialize
   the submodule:

   ```bash
   python build_tools/fetch_sources.py
   ```

   You should see a `git submodule update --init` call and more in the logs.

1. Check that the submodule was checked out:

   ```console
   $ ls math-libs/BLAS/rocRoller

   client/                   docker/                    LICENSE.md   requirements.txt
   cmake/                    docs/                      next-cmake/  scripts/
   CMakeLists.txt            extern/                    patches/     test/
   codeql/                   GPUArchitectureGenerator/  pytest.ini   valgrind.supp
   CppCheckSuppressions.txt  lib/                       README.md
   ```

The submodule should be fully configured now!

> [!TIP]
> Now that the source directory is populated, see
> [TheRock Build System Manual: Adding Sub-Projects](./build_system.md#adding-sub-projects)
> to use it from the CMake build system.
