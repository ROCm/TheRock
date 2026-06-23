# External Repo Integration

TheRock's multi-arch CI can be driven by an external repository — a repo
that contains one or more ROCm components and wants to build and test them
against TheRock's full stack without maintaining a fork.

Currently supported external repos: `rocm-libraries`, `rocm-systems`,
`rocgdb`. Adding a new repo requires a one-time registration in
[`detect_external_repo_config.py`](../../build_tools/github_actions/detect_external_repo_config.py).

## How it works

The caller invokes TheRock's reusable
[`setup_multi_arch.yml`](../../.github/workflows/setup_multi_arch.yml)
workflow with an `external_repo` JSON input. TheRock checks out the caller's
source at the specified ref, substitutes it for the corresponding submodule,
and runs the build and test pipeline.

```yaml
# In the external repo's workflow file:
jobs:
  setup:
    uses: ROCm/TheRock/.github/workflows/setup_multi_arch.yml@<sha>
    with:
      external_repo: >-
        {
          "repository": "ROCm/ROCgdb",
          "ref": "${{ github.sha }}",
          "extra_cmake_options": "-DTHEROCK_USE_EXTERNAL_ROCGDB=ON",
          "skip_packaging": true
        }
```

## `external_repo` JSON fields

| Field                 | Required | Description                                                                                                             |
| --------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------- |
| `repository`          | Yes      | GitHub repo in `owner/name` format (e.g. `"ROCm/ROCgdb"`). Must match a known repo name (case-insensitive).             |
| `ref`                 | Yes      | Git SHA, branch, or tag to check out.                                                                                   |
| `extra_cmake_options` | No       | CMake flags forwarded to the configure step of stages that own the component (e.g. `-DTHEROCK_USE_EXTERNAL_ROCGDB=ON`). |
| `skip_packaging`      | No       | `true` to suppress DEB/RPM packages, Python wheels, and PyTorch wheel jobs. Defaults to `false`.                        |

## Stage scoping

When an external repo is provided, TheRock automatically scopes the build
to only the stages that are relevant to the component:

1. `detect_external_repo_config.py` looks up which submodule(s) the repo
   replaces (e.g. `rocgdb` → submodule `rocgdb`).
1. It calls `StageImpactAnalyzer.required_stages_for_component()` to walk
   the build graph *upstream* from the owning stage, collecting every stage
   whose artifacts the owning stage depends on.
1. The resulting `build_stages` list is embedded in `external_repo_config`
   and consumed by the build workflows.

**Example:** ROCgdb owns the `debug-tools` stage, which depends on
`compiler-runtime`. So only `compiler-runtime` and `debug-tools` run;
all other stages (`math-libs`, `storage-libs`, `media-libs`, etc.) are
skipped.

`extra_cmake_options` are similarly scoped: flags are only injected into
the configure step of stages listed in `build_stages`, not every stage.

An empty `build_stages` (returned for unknown submodules) means no
restriction — all stages run as normal.

## Packaging suppression

By default, multi-arch CI also builds DEB/RPM packages, Python wheels,
and PyTorch wheels. For external component builds these are almost never
needed and add significant runtime. Setting `"skip_packaging": true` in
the `external_repo` JSON disables all three packaging job groups.

```yaml
external_repo: '{"repository":"ROCm/ROCgdb","ref":"${{ github.sha }}","skip_packaging":true}'
```

This sets `build_native_linux`, `build_python_packages`, and
`build_pytorch` to `false` in the generated `LinuxBuildConfig`, which
gates the corresponding jobs in `multi_arch_ci_linux.yml` and
`multi_arch_ci_windows.yml`.

## Adding a new external repo

1. **Register the repo** in `REPO_CONFIGS` in
   [`detect_external_repo_config.py`](../../build_tools/github_actions/detect_external_repo_config.py):

   ```python
   "my-component": {
       "cmake_source_var": "THEROCK_MY_COMPONENT_SOURCE_DIR",
       "submodule_path": "path/to/submodule/in/therock",
       "skip_submodules": ["my-component"],
   },
   ```

   - `cmake_source_var`: CMake cache variable TheRock uses to locate the
     external source tree.
   - `submodule_path`: path of the submodule inside the TheRock checkout
     that will be replaced by the external source.
   - `skip_submodules`: list of submodule names used to derive the required
     build stages from `BUILD_TOPOLOGY.toml`.

1. **Wire up the CMake variable** in the component's TheRock integration
   (e.g. `cmake/therock_my_component.cmake`) to consume
   `THEROCK_MY_COMPONENT_SOURCE_DIR` when set.

1. **Create the caller workflow** in the external repo (see the example
   above). Reference a pinned SHA of TheRock's
   `setup_multi_arch.yml` for reproducibility.

## Reference

- [`setup_multi_arch.yml`](../../.github/workflows/setup_multi_arch.yml) — reusable workflow entry point
- [`detect_external_repo_config.py`](../../build_tools/github_actions/detect_external_repo_config.py) — repo registration and config detection
- [`configure_multi_arch_ci.py`](../../build_tools/github_actions/configure_multi_arch_ci.py) — build config generation (reads `EXTERNAL_REPO_JSON`)
- [`stage_impact.py`](../../build_tools/github_actions/stage_impact.py) — `StageImpactAnalyzer.required_stages_for_component()`
- [`BUILD_TOPOLOGY.toml`](../../BUILD_TOPOLOGY.toml) — stage definitions and artifact dependency graph
- [ci_overview.md](ci_overview.md) — general CI architecture
- [ci_behavior_manipulation.md](ci_behavior_manipulation.md) — prebuilt stages, labels, trigger behavior
