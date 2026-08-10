# Multi-Arch CI Integration Guide

This guide explains how to integrate TheRock's multi-arch CI pipeline into
external ROCm repositories (e.g., `rocm-libraries`, `rocm-systems`).

## Overview

TheRock provides reusable workflows that external repos can call to get:

- Sharded, multi-stage builds (foundation -> compiler-runtime -> math-libs)
- Per-architecture testing on real GPU hardware
- Automatic stage reuse from compatible baselines
- Support for release, ASAN, and nightly builds

## Quick Start

### Minimal Workflow

```yaml
# .github/workflows/therock-multi-arch-ci.yml
name: TheRock Multi-Arch CI

on:
  pull_request:
  workflow_dispatch:

permissions:
  contents: read
  actions: read

jobs:
  setup:
    uses: ROCm/TheRock/.github/workflows/setup_multi_arch.yml@main
    with:
      build_variant: "release"
      linux_amdgpu_families: "gfx94X,gfx950"
      repository: ROCm/TheRock
      ref: main
      external_repo: '{"repository":"${{ github.repository }}","ref":"${{ github.sha }}"}'

  linux_build_and_test:
    needs: setup
    if: needs.setup.outputs.linux_build_config != ''
    uses: ROCm/TheRock/.github/workflows/multi_arch_ci_linux.yml@main
    secrets: inherit
    with:
      build_config: ${{ needs.setup.outputs.linux_build_config }}
      test_labels: ${{ needs.setup.outputs.linux_test_labels }}
      rocm_package_version: ${{ needs.setup.outputs.rocm_package_version }}
      test_type: ${{ needs.setup.outputs.test_type }}
      external_repo_config: ${{ needs.setup.outputs.external_repo_config }}
      repository: ROCm/TheRock
      ref: ${{ needs.setup.outputs.ref }}
    permissions:
      contents: read
      id-token: write
```

## Full Example (rocm-libraries)

```yaml
name: TheRock Multi-Arch CI

on:
  pull_request:
    types: [opened, synchronize, reopened]
  workflow_dispatch:
    inputs:
      linux_amdgpu_families:
        type: string
        description: "GPU families (e.g., gfx94X,gfx120X)"
        default: ""
      linux_test_labels:
        type: string
        description: "Test labels (e.g., test:rocprim,test:hipcub)"
        default: ""
      prebuilt_stages:
        type: string
        default: ""
        description: "Stages to skip (or 'all'); requires baseline_run_id"
      baseline_run_id:
        type: string
        default: ""
        description: "Run ID to copy prebuilt artifacts from"
      therock_ref_override:
        type: string
        default: ""
        description: "Explicit TheRock ref (branch/tag/SHA)"

permissions:
  contents: read
  actions: read

concurrency:
  group: ${{ github.workflow }}-${{ github.event.number || github.sha }}
  cancel-in-progress: true

jobs:
  # Detect which projects changed for targeted testing
  configure:
    name: Configure Changed Projects
    runs-on: ubuntu-24.04
    outputs:
      changed_projects: ${{ steps.configure.outputs.changed_projects }}
      run_all_tests: ${{ steps.configure.outputs.run_all_tests }}
      skip_tests: ${{ steps.configure.outputs.skip_tests }}
    steps:
      - name: Checkout repository config
        uses: actions/checkout@v4
        with:
          sparse-checkout: .github/repos-config.json
          sparse-checkout-cone-mode: false

      - name: Checkout TheRock for configure script
        uses: actions/checkout@v4
        with:
          repository: ROCm/TheRock
          ref: main
          path: _therock
          sparse-checkout: build_tools/github_actions
          sparse-checkout-cone-mode: true

      - name: Configure CI
        id: configure
        env:
          GITHUB_TOKEN: ${{ github.token }}
        run: |
          python3 _therock/build_tools/github_actions/configure_external_repo_ci.py \
            --event-name "${{ github.event_name }}" \
            --github-repo "${{ github.repository }}" \
            --base-sha "${{ github.event.pull_request.base.sha }}" \
            --head-sha "${{ github.event.pull_request.head.sha }}" \
            --config-path ".github/repos-config.json"

  # Pin TheRock to merge-base time for stable PR builds
  resolve_therock_ref:
    name: Resolve TheRock ref
    runs-on: ubuntu-24.04
    outputs:
      therock_ref: ${{ steps.resolve.outputs.therock_ref }}
    steps:
      - uses: actions/checkout@v4
        with:
          sparse-checkout: .github
          sparse-checkout-cone-mode: true

      - run: pip install requests

      - name: Resolve TheRock ref
        id: resolve
        env:
          GITHUB_TOKEN: ${{ github.token }}
          BASE_SHA: ${{ github.event.pull_request.base.sha }}
          HEAD_SHA: ${{ github.event.pull_request.head.sha }}
          THEROCK_REF_OVERRIDE: ${{ inputs.therock_ref_override || '' }}
        run: python .github/scripts/resolve_therock_ref.py

  setup:
    needs: [configure, resolve_therock_ref]
    uses: ROCm/TheRock/.github/workflows/setup_multi_arch.yml@main
    with:
      build_variant: "release"
      linux_amdgpu_families: ${{ inputs.linux_amdgpu_families || 'gfx94X,gfx950' }}
      prebuilt_stages: ${{ inputs.prebuilt_stages || '' }}
      baseline_run_id: ${{ inputs.baseline_run_id || '' }}
      stage_reuse_mode: reuse-stage
      repository: ROCm/TheRock
      ref: ${{ needs.resolve_therock_ref.outputs.therock_ref }}
      external_repo: '{"repository":"${{ github.repository }}","ref":"${{ github.sha }}"}'

  linux_build_and_test:
    name: Linux::${{ fromJSON(needs.setup.outputs.linux_build_config || '{}').build_variant_label || 'skip' }}
    needs: [configure, setup]
    if: >-
      needs.setup.outputs.linux_build_config != '' &&
      needs.setup.outputs.enable_build_jobs == 'true' &&
      needs.configure.outputs.skip_tests != 'true'
    uses: ROCm/TheRock/.github/workflows/multi_arch_ci_linux.yml@main
    secrets: inherit
    with:
      build_config: ${{ needs.setup.outputs.linux_build_config }}
      test_labels: ${{ needs.setup.outputs.linux_test_labels }}
      rocm_package_version: ${{ needs.setup.outputs.rocm_package_version }}
      test_type: ${{ needs.setup.outputs.test_type }}
      external_repo_config: ${{ needs.setup.outputs.external_repo_config }}
      # Empty string = run all tests; otherwise run only changed projects
      changed_projects: ${{ needs.configure.outputs.run_all_tests == 'true' && '' || needs.configure.outputs.changed_projects }}
      repository: ROCm/TheRock
      ref: ${{ needs.setup.outputs.ref }}
    permissions:
      contents: read
      id-token: write

  multi_arch_ci_summary:
    name: Multi-Arch CI Summary
    if: always()
    needs: [setup, linux_build_and_test]
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
        with:
          repository: "ROCm/TheRock"
          ref: ${{ needs.setup.outputs.ref }}
          sparse-checkout: build_tools/github_actions
          sparse-checkout-cone-mode: true

      - name: Evaluate workflow results
        env:
          GITHUB_TOKEN: ${{ github.token }}
        run: |
          python build_tools/github_actions/workflow_summary.py \
            --needs-json '${{ toJSON(needs) }}'
```

## Key Concepts

### setup_multi_arch.yml Inputs

| Input                    | Description                                                                |
| ------------------------ | -------------------------------------------------------------------------- |
| `build_variant`          | `release`, `asan`, or `host-asan`                                          |
| `linux_amdgpu_families`  | Comma-separated GPU families (e.g., `gfx94X,gfx950,gfx120X`)               |
| `windows_amdgpu_families`| Same for Windows                                                           |
| `linux_test_labels`      | Filter tests to specific labels (e.g., `test:rocprim,test:hipcub`)         |
| `prebuilt_stages`        | Skip building these stages; copy from `baseline_run_id`                    |
| `baseline_run_id`        | Workflow run ID to copy prebuilt artifacts from                            |
| `stage_reuse_mode`       | `dry-run` (report only) or `reuse-stage` (actually reuse)                  |
| `repository`             | TheRock repository (usually `ROCm/TheRock`)                                |
| `ref`                    | TheRock ref to build against                                               |
| `external_repo`          | JSON with external repo info (see below)                                   |

### setup_multi_arch.yml Outputs

| Output                | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `linux_build_config`  | JSON with Linux build configuration                            |
| `windows_build_config`| JSON with Windows build configuration                          |
| `enable_build_jobs`   | Whether to run build jobs                                      |
| `linux_test_labels`   | Resolved test labels                                           |
| `rocm_package_version`| Computed ROCm package version                                  |
| `test_type`           | Test type (`quick`, `standard`, `comprehensive`, `full`)       |
| `ref`                 | Resolved TheRock commit SHA (use this downstream)              |
| `external_repo_config`| JSON config for external repo checkout                         |

### external_repo JSON

Pass your repository info so TheRock can check out your code during builds:

```yaml
external_repo: '{"repository":"${{ github.repository }}","ref":"${{ github.sha }}"}'
```

## Adding Custom GPU Family Configuration

For new or experimental GPUs, use `family_overrides` in `external_repo`:

```yaml
# Example: rocm-systems adding gfx125X with custom test runner
external_repo: >-
  {
    "repository": "${{ github.repository }}",
    "ref": "${{ github.sha }}",
    "family_overrides": {
      "gfx125x": {
        "linux": {
          "test-runs-on": "linux-mi455-gpu-rocm",
          "test_labels_for_family": ["test:hip-tests", "test:rocrtst"],
          "fetch-gfx-targets": ["gfx1250"]
        }
      }
    }
  }
```

### family_overrides Fields

| Field                   | Description                                                     |
| ----------------------- | --------------------------------------------------------------- |
| `test-runs-on`          | GitHub runner label for test jobs (empty string = skip tests)   |
| `test_labels_for_family`| Limit tests to specific labels for this GPU family              |
| `fetch-gfx-targets`     | Specific GPU targets to fetch (e.g., `gfx1250` for `gfx125X`)   |

This lets you configure custom runners for hardware not yet in TheRock's default
configuration, or limit which tests run on experimental hardware.

## ASAN Builds

For Address Sanitizer builds, set `build_variant: "asan"`:

```yaml
setup:
  uses: ROCm/TheRock/.github/workflows/setup_multi_arch.yml@main
  with:
    build_variant: "asan"
    linux_amdgpu_families: 'gfx94X,gfx950,gfx125X'
    repository: ROCm/TheRock
    ref: main
    external_repo: '{"repository":"${{ github.repository }}","ref":"${{ github.sha }}"}'
```

TheRock automatically selects `host-asan` for push events (faster, host-only)
and full `asan` for workflow_dispatch (comprehensive device-side instrumentation).

## Nightly Builds

For scheduled nightly runs:

```yaml
on:
  schedule:
    - cron: "0 7 * * *"  # 7 AM UTC daily
  workflow_dispatch:
    inputs:
      therock_ref_override:
        type: string
        default: ""

jobs:
  setup:
    uses: ROCm/TheRock/.github/workflows/setup_multi_arch.yml@main
    with:
      build_variant: "release"
      linux_amdgpu_families: 'gfx94X,gfx950'
      windows_amdgpu_families: 'gfx1151'
      repository: ROCm/TheRock
      # Use override if provided, otherwise live tip of main
      ref: ${{ inputs.therock_ref_override || 'main' }}
      external_repo: '{"repository":"${{ github.repository }}","ref":"${{ github.sha }}"}'
```

## TheRock Ref Resolution

### The Problem

If your PR workflow always uses `ROCm/TheRock@main`, the TheRock version can
change mid-PR as main advances, causing inconsistent builds.

### The Solution: resolve_therock_ref.py

Copy this script to `.github/scripts/resolve_therock_ref.py`:

```python
#!/usr/bin/env python3
"""Resolve the ROCm/TheRock commit a multi-arch CI run should build against.

Policy:
- pull_request: Pin to TheRock@main as it existed at the PR's merge-base time.
  Stays frozen while you push commits; advances when you merge/rebase base.
- workflow_dispatch/schedule/push: Use live tip of main.
- therock_ref_override: Manual pin takes precedence.
"""
```

See the full implementation in
[rocm-libraries/.github/scripts/resolve_therock_ref.py](https://github.com/ROCm/rocm-libraries/blob/develop/.github/scripts/resolve_therock_ref.py).

### How It Works

1. For PRs, computes the merge-base between your branch and the base branch
2. Finds the TheRock@main commit at or before that merge-base timestamp
3. Uses that commit for the entire PR lifecycle
4. Only advances when you sync your branch with the base

This keeps the ROCm build stable while you author and debug your PR.

## Change Detection Flow

The `configure_external_repo_ci.py` script determines what changed:

```
PR opened/synchronized
         │
         ▼
┌─────────────────────┐
│ Get modified files  │
│ via GitHub API      │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐     Yes    ┌──────────────────┐
│ CI files changed?   │───────────▶│ run_all_tests    │
│ (.github/workflows) │            │ = true           │
└─────────────────────┘            └──────────────────┘
         │ No
         ▼
┌─────────────────────┐     Yes    ┌──────────────────┐
│ Only docs/skippable │───────────▶│ skip_tests       │
│ files changed?      │            │ = true           │
└─────────────────────┘            └──────────────────┘
         │ No
         ▼
┌─────────────────────┐
│ Match changed paths │
│ to repos-config.json│
│ project prefixes    │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Output:             │
│ changed_projects    │
│ = "projects/rocblas,│
│    projects/hipblas"│
└─────────────────────┘
```

### repos-config.json

Your repo needs a config file mapping directories to projects:

```json
{
  "repositories": [
    {"name": "rocblas", "category": "projects", "url": "...", "branch": "..."},
    {"name": "hipblas", "category": "projects", "url": "...", "branch": "..."}
  ]
}
```

## Testing Your Changes in TheRock

When modifying TheRock's reusable workflows:

### 1. Point External Repo to Your Branch

```yaml
# In your external repo's workflow, temporarily change:
uses: ROCm/TheRock/.github/workflows/setup_multi_arch.yml@main
# To:
uses: ROCm/TheRock/.github/workflows/setup_multi_arch.yml@users/yourname/feature

# And set ref to your branch:
ref: users/yourname/feature
```

### 2. Use workflow_dispatch with therock_ref_override

If the external repo supports it:

1. Go to Actions > TheRock Multi-Arch CI
2. Click "Run workflow"
3. Set `therock_ref_override` to your TheRock branch/SHA

### 3. Test in TheRock Directly

TheRock's own CI runs the same multi-arch pipeline. Make changes to TheRock,
push to a branch, and trigger the CI there.

## Required Permissions

```yaml
permissions:
  contents: read    # Checkout code
  actions: read     # Required for cross-repo workflow calls
  id-token: write   # Required for AWS artifact uploads
```

Note: `actions: read` is required when calling reusable workflows in a different
repository.

## Troubleshooting

### "Resource not accessible by integration"

Add `actions: read` to your workflow permissions.

### Tests not running on your GPU

Check that your GPU family is in `linux_amdgpu_families` and that TheRock's
`therock-ci-config` has test runners configured for it. For new GPUs, use
`family_overrides`.

### Stage reuse not working

Ensure `stage_reuse_mode: reuse-stage` is set. Check the workflow summary for
reuse decisions. Baselines must be:
- From the same repository
- Within `stage_reuse_max_age_hours` (default 72)
- From an ancestor commit

### TheRock ref keeps changing

Implement `resolve_therock_ref.py` to pin TheRock at merge-base time for PRs.
