# Patches for TheRock

> [!WARNING]
> New patches are STRONGLY DISCOURAGED.

## Working with patches

### Applying patches with `fetch_sources.py`

The [`build_tools/fetch_sources.py`](/build_tools/fetch_sources.py) script
checks out git submodules and then applies sets of patches from this directory:

```bash
# Typical usage with default arguments
#   (--apply-patches defaults to True)
#   (--patch-tag     defaults to "amd-mainline")
python build_tools/fetch_sources.py

# Equivalent to:
python build_tools/fetch_sources.py --apply-patches --patch-tag amd-mainline
```

This command:

1. Checks out the specified submodules
1. Looks for patch files in `patches/<patch-tag>/<project>/*.patch`
1. Applies patches using `git am --whitespace=nowarn` (as user `therockbot`)

To skip applying patches:

```bash
python build_tools/fetch_sources.py --no-apply-patches
```

<details>
<summary>Technical details: version tracking and git configuration</summary>

After applying patches, the script:

- Generates a `.smrev` file (e.g., `.rocm-libraries.smrev`) containing the
  submodule URL and a hash of the applied patches for version tracking
- Sets `git update-index --skip-worktree` on the patched submodule to mark it
  as modified. This allows commands like `git status` from the base repository
  to report no differences even though patched submodules contain new commits.

</details>

### Saving patches with `save_patches.sh`

Use [`build_tools/save_patches.sh`](/build_tools/save_patches.sh) to save local
commits for a given submodule as patch files:

```bash
# Syntax
./build_tools/save_patches.sh <base_tag> <project_name> <patch_subdir>

# Example: Save commits to hipBLASLt since the m/amd-mainline tag
./build_tools/save_patches.sh m/amd-mainline hipBLASLt amd-mainline
```

This script:

1. Removes existing `*.patch` files in `patches/<patch_subdir>/<project_name>/`
1. Uses `git format-patch` to generate patches for all commits since `<base_tag>`
1. Saves the patches to `patches/<patch_subdir>/<project_name>/`
   - When working with an already-patched submodule, this script will
     regenerate the same patch files (or updated versions if you've modified
     commits).

### Using patches from rocm-libraries, rocm-systems, and other repositories

Several external repositories like
[`rocm-libraries`](https://github.com/ROCm/rocm-libraries) and
[`rocm-systems`](https://github.com/ROCm/rocm-systems) use the build system from
TheRock in their own CI workflows. If there are patches in TheRock for those
submodules, then those CI workflows may need to handle them carefully.

**The typical CI pattern**:

1. The external repository checks out its own code at HEAD
1. Checks out TheRock into a subdirectory (e.g., `./TheRock`)
1. Fetches all _other_ dependencies via TheRock's `fetch_sources.py` (excluding
   the external repo itself)
1. Applies patches from `TheRock/patches/amd-mainline/<repo>/` to the external repository
1. Builds using TheRock with the patched external repository as a source override

**Example from rocm-libraries**:

```yml
      # (1)
      - name: "Checking out repository for rocm-libraries"
        uses: actions/checkout@08c6903cd8c0fde910a37f88322edcfb5dd907a8 # v5.0.0

      # (2)
      - name: Checkout TheRock repository
        uses: actions/checkout@08c6903cd8c0fde910a37f88322edcfb5dd907a8 # v5.0.0
        with:
          repository: "ROCm/TheRock"
          path: TheRock
          ref: a1f6b57cc31890c05ab0094212ae0b269765db8e # 2025-11-25 commit

      # (3)
      - name: Fetch sources
        run: |
          ./TheRock/build_tools/fetch_sources.py --jobs 12 \
              --no-include-rocm-libraries --no-include-ml-frameworks

      # (4)
      - name: Patch rocm-libraries
        run: |
          # Remove patches here if they cannot be applied cleanly, and they have not been deleted from TheRock repo
          # rm ./TheRock/patches/amd-mainline/rocm-libraries/*.patch
          git -c user.name="therockbot" -c "user.email=therockbot@amd.com" am \
              --whitespace=nowarn ./TheRock/patches/amd-mainline/rocm-libraries/*.patch

      # (5)
      - name: Configure Projects
        env:
          extra_cmake_options: "-DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR=../"
          BUILD_DIR: build
        run: python3 TheRock/build_tools/github_actions/build_configure.py
      - name: Build therock-dist
        run: cmake --build TheRock/build --target therock-dist
```

Example full workflows:

- [rocm-libraries - `.github/workflows/therock-ci-linux.yml`](https://github.com/ROCm/rocm-libraries/blob/develop/.github/workflows/therock-ci-linux.yml)
- [rocm-systems - `.github/workflows/therock-ci-linux.yml`](https://github.com/ROCm/rocm-systems/blob/develop/.github/workflows/therock-ci-linux.yml)

#### Resolving conflicts with patches

> [!IMPORTANT]
> If a patch does not apply, such as when another commit modifies a file that
> the patch modifies, the workflow will fail.
>
> - If the conflicting commit is itself equivalent to the patch, then the patch
>   can be deleted via the (commented out) `rm ...` line. When TheRock picks up
>   the equivalent commit and the external repository in turns picks up a new
>   the commit `ref` for TheRock, the `rm ...` line can be removed again.
> - If the conflicting commit is NOT equivalent to the patch, for example if it
>   modifies unrelated lines in one of the patched files, then a different
>   resolution is needed. **This is why patches are expensive and should be
>   avoided at all costs - they can block regular project development!**

## Rules for creating patches

**Remember**: New patches are **STRONGLY DISCOURAGED**. They create maintenance
burden and can block development in external repositories.

### Prefer upstream fixes

If a commit in a repository like
[`rocm-libraries`](https://github.com/ROCm/rocm-libraries) or
[`rocm-systems`](https://github.com/ROCm/rocm-systems) causes issues:

1. **First choice**: Revert the problematic commit in that repository, then
   pick up the fix via regular submodule updates
1. **Last resort**: Create a patch in TheRock only if the upstream revert is
   not feasible

Supporting legacy build systems or closed-source development processes is NOT
sufficient justification for breaking public builds and tests.

### When patches are acceptable

Patches may be used for:

- **Third-party compatibility**: Adapting third-party dependencies that lack
  active maintenance or need project-specific changes that are unlikely to be
  accepted upstream (see [`patches/third-party/`](./third-party/))
- **Temporary workarounds**: Short-term fixes for critical issues while waiting
  for commits to be merged upstream. Any commits carried as patches
  **must** be on a path to being reviewed and accepted into their respective
  repositories.

Patches should NOT be used for:

- Working around issues that can be fixed by reverting problematic commits in
  the source repository
  - Public builds and releases take priority over closed-source builds and
    releases
    builds
- Adding features that should be developed upstream
- Long-term divergence from upstream projects

### Commit message requirements

Patches are held to a higher commit message standard than standard commits.
Commit messages for patches should include:

1. **Why the patch exists**: Explain the problem being solved and why a patch
   is necessary instead of an upstream fix
1. **Upstream tracking**: Link to related upstream issues, pull requests, or
   discussions if applicable
1. **Reproduction info**: For bug fixes, include error messages or reproduction
   steps
