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
1. Only applies patches for projects that were included in the fetch

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

Use [`build_tools/save_patches.sh`](/build_tools/save_patches.sh) to save local commits as patch files:

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

**Typical workflow**: When working with already-patched submodules, this script will
regenerate the same patch files (or updated versions if you've modified commits).
The "remove existing patch files" step ensures the patch directory stays clean and
matches your current commit history.

### Using patches from rocm-libraries, rocm-systems, and other repositories

External repositories like `rocm-libraries` and `rocm-systems` use TheRock's patches
in their CI workflows to test changes against TheRock's build system. This creates a
feedback loop where repositories can validate that their changes work with TheRock
before merging.

**The typical CI pattern**:

1. The external repository checks out its own code at HEAD
1. Checks out TheRock into a subdirectory (e.g., `./TheRock`)
1. Fetches all other dependencies via TheRock's `fetch_sources.py` (excluding the external repo itself)
1. Applies patches from `TheRock/patches/amd-mainline/<repo>/` to the external repository
1. Builds using TheRock with the patched external repository as a source override

**Example from rocm-libraries**:

```yml
- name: Patch rocm-libraries
  run: |
    # Remove patches here if they cannot be applied cleanly, and they have not been deleted from TheRock repo
    # rm ./TheRock/patches/amd-mainline/rocm-libraries/*.patch
    git -c user.name="therockbot" -c "user.email=therockbot@amd.com" am --whitespace=nowarn ./TheRock/patches/amd-mainline/rocm-libraries/*.patch
```

See the full workflows:

- [rocm-libraries/.github/workflows/therock-ci-linux.yml](https://github.com/ROCm/rocm-libraries/blob/develop/.github/workflows/therock-ci-linux.yml)
- [rocm-systems/.github/workflows/therock-ci-linux.yml](https://github.com/ROCm/rocm-systems/blob/develop/.github/workflows/therock-ci-linux.yml)

**Important**: The commented `rm` line is a safety valve. If a patch cannot be applied
cleanly (e.g., after upstream changes), CI maintainers can uncomment this to remove
problematic patches, allowing the build to proceed while the patch conflict is resolved.

## Rules for creating patches

- If a commit in a repository like rocm-libraries or rocm-systems causes issues,
  the first thought should be to revert that commit in that repository and then
  pick up the latest code from that project via regular submodule updates ("roll-ups"),
  rather than cherry-pick the revert via a local patch in TheRock
  - Supporting legacy build and release systems or other closed source development
    processes is NOT sufficient justification for breaking public builds and tests
- Add context to the commit message for
