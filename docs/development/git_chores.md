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
     --projects clr \
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
