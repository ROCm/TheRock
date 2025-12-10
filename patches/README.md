# Patches for TheRock

> [!WARNING]
> New patches are STRONGLY DISCOURAGED.

## Working with patches

The [`build_tools/fetch_sources.py`](/build_tools/fetch_sources.py) script
checks out git submodules and then applies sets of patches from this directory.

TODO: explain how `--apply-patches` in that script works

TODO: explain how to use `build_tools/save_patches.sh`

### Using patches from rocm-libraries, rocm-systems, and other repositories

TODO: explain the code in rocm-libraries/.github/workflows/therock-ci-linux.yml
TODO: link to https://github.com/ROCm/rocm-libraries/blob/develop/.github/workflows/therock-ci-linux.yml

```yml
- name: Patch rocm-libraries
  run: |
    # Remove patches here if they cannot be applied cleanly, and they have not been deleted from TheRock repo
    # rm ./TheRock/patches/amd-mainline/rocm-libraries/*.patch
    git -c user.name="therockbot" -c "user.email=therockbot@amd.com" am --whitespace=nowarn ./TheRock/patches/amd-mainline/rocm-libraries/*.patch

```

## Rules for creating patches

- If a commit in a repository like rocm-libraries or rocm-systems causes issues,
  the first thought should be to revert that commit in that repository and then
  pick up the latest code from that project via regular submodule updates ("roll-ups"),
  rather than cherry-pick the revert via a local patch in TheRock
  - Supporting legacy build and release systems or other closed source development
    processes is NOT sufficient justification for breaking public builds and tests
- Add context to the commit message for
