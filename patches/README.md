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

#### Automated patch application (CI workflows)

External repositories use TheRock's reusable `ci.yml` workflow via `workflow_call`.
Patches are applied automatically by `fetch_sources.py`:

```yml
jobs:
  build:
    uses: ROCm/TheRock/.github/workflows/ci.yml@some-ref
    with:
      external_source_checkout: true
      repository_override: ${{ github.repository }}
      # Other inputs as needed (projects, amdgpu_families, etc.)
```

**How patches are applied automatically**:

1. TheRock's `ci.yml` workflow checks out both TheRock and the external repository
1. `fetch_sources.py` is called with `EXTERNAL_SOURCE_CHECKOUT=true` and `EXTERNAL_SOURCE_PATH` set
1. Patches from `patches/amd-mainline/<repo>/` are automatically applied to the external repository
1. Build proceeds with the patched external repository

**Skipping specific patches during removal** (see [Removing Patches](#removing-patches)):

When removing patches, external repositories can temporarily skip specific patches using the `skip_patches` input:

```yml
jobs:
  build:
    uses: ROCm/TheRock/.github/workflows/ci.yml@some-ref
    with:
      external_source_checkout: true
      repository_override: ${{ github.repository }}
      skip_patches: "0001-workaround.patch,0002-temp-fix.patch"  # Comma-separated filenames
```

This allows external repositories to merge fixes before TheRock removes the corresponding patch files.

#### Manual patch application (local development)

For local development, you can manually apply patches using environment variables:

```bash
# Set up environment variables
export EXTERNAL_SOURCE_CHECKOUT=true
export EXTERNAL_SOURCE_PATH=/path/to/your/rocm-libraries  # Your local checkout path

# Fetch sources and apply patches
cd TheRock
python build_tools/fetch_sources.py --no-include-rocm-libraries

# Verify patches were applied
cd /path/to/your/rocm-libraries
git log -3  # Should show patch commits by therockbot

# Build
cd TheRock
cmake -B build -DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR=/path/to/your/rocm-libraries
cmake --build build
```

To skip specific patches locally:

```bash
export SKIP_PATCHES="0001-workaround.patch,0002-temp-fix.patch"
python build_tools/fetch_sources.py --no-include-rocm-libraries
```

#### Under the hood: what `fetch_sources.py` does

When applying patches, `fetch_sources.py` runs the following git command for each project:

```bash
git -c user.name="therockbot" \
    -c user.email="therockbot@amd.com" \
    am --whitespace=nowarn \
    patches/<patch-tag>/<project>/*.patch
```

- `user.name` and `user.email`: Patch commits are attributed to "therockbot"
- `am --whitespace=nowarn`: Apply mailbox patches, ignoring whitespace warnings
- `GIT_COMMITTER_DATE`: Set to fixed date for reproducible builds

Patches listed in `SKIP_PATCHES` are filtered out before running `git am`.

#### Resolving conflicts with patches

> [!IMPORTANT]
> If a patch does not apply, such as when another commit modifies a file that
> the patch modifies, the workflow will fail.
>
> - If the conflicting commit is itself equivalent to the patch, use the
>   `skip_patches` input to temporarily skip the patch (see [Removing Patches](#removing-patches))
> - If the conflicting commit is NOT equivalent to the patch, for example if it
>   modifies unrelated lines in one of the patched files, then the patch needs to
>   be regenerated or removed. **This is why patches are expensive and should be
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

## Removing Patches

Patches should be removed as soon as the underlying fix is available upstream or when they become obsolete.

### For TheRock submodules (llvm-project, third-party libraries)

Use a single PR approach:

1. **Update the submodule** to pick up the upstream commit that replaces the patch
1. **Delete the patch file(s)** from `patches/<patch-tag>/<project>/`
1. **Test the build** without the patch:
   ```bash
   python build_tools/fetch_sources.py
   cmake -B build
   cmake --build build
   ```
1. **Commit both changes** (submodule update + patch deletion) together
1. **Open a PR** with a reference to the upstream commit that made the patch obsolete

### For external repositories (rocm-libraries, rocm-systems)

External repositories require a **three-step coordinated process** using the `skip_patches` input to avoid the chicken-and-egg problem where:

- TheRock can't remove the patch until external repo has the fix
- External repo CI can't pass until TheRock removes the patch

**Step 1: External repo PR (merge first)**

1. Merge the upstream fix that makes the patch obsolete into the external repository
1. Add `skip_patches` to the workflow file to temporarily skip the patch during CI:

```yml
jobs:
  build:
    uses: ROCm/TheRock/.github/workflows/ci.yml@some-ref
    with:
      external_source_checkout: true
      repository_override: ${{ github.repository }}
      skip_patches: "0001-workaround.patch"  # Add the patch filename(s) to skip
```

3. Verify CI passes (the patch is skipped automatically by `fetch_sources.py`)
1. **Merge this PR** - the external repo now has the fix and can build with or without the patch

**Step 2: TheRock PR (merge second)**

1. **Delete the patch file(s)** from `patches/amd-mainline/<project>/`
1. **Document in the PR description**:
   - Which external repo commit made the patch obsolete
   - Link to the external repo PR from Step 1
1. Test by having the external repo CI reference your TheRock branch
1. **Merge this PR** - TheRock no longer has the patch file

**Step 3: External repo cleanup PR (merge third)**

1. Update the TheRock ref in the external repo workflow to point to the merged commit from Step 2
1. **Remove the `skip_patches` line** from the workflow (patch file no longer exists)
1. Verify CI still passes
1. **Merge this PR** - cleanup complete

### Testing patch removal locally

**For submodules:**

```bash
# Update submodule and test without patch
git submodule update --init <project>
python build_tools/fetch_sources.py
# Verify no patch application messages in output
cmake -B build
cmake --build build
```

**For external repos:**

```bash
# Test with skip_patches
export SKIP_PATCHES="0001-workaround.patch"
export EXTERNAL_SOURCE_CHECKOUT=true
export EXTERNAL_SOURCE_PATH=/path/to/your/rocm-libraries
python build_tools/fetch_sources.py --no-include-rocm-libraries

# Verify patch was skipped in output
cd /path/to/your/rocm-libraries
git log -3  # Should NOT show the skipped patch commit
```
