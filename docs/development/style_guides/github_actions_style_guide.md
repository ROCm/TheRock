# GitHub Actions style guide

## Style guidelines

### Pin action `uses:` versions to commit SHAs

Pin actions in
[`jobs.<job_id>.steps[*].uses`](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idstepsuses)
to specific commit SHAs for security and reproducibility. Do not use release
tags like `@v6` or branch names like `@main` as these can change outside of our
control.

Benefits:

- **Security:** Prevents malicious code injection via tag/branch updates
- **Reproducibility:** Ensures workflows behave consistently over time
- **Transparency:** Clear which exact version is being used
- **Dependabot compatibility:** Works seamlessly with automatic updates

✅ **Preferred:**

```yaml
- uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683  # v4.2.2
- uses: docker/setup-buildx-action@c47758b77c9736f4b2ef4073d4d51994fabfe349  # v3.7.1
```

> [!TIP]
> We use
> [Dependabot](https://docs.github.com/en/code-security/dependabot/working-with-dependabot/keeping-your-actions-up-to-date-with-dependabot)
> to automatically update pinned actions while maintaining security.
>
> Dependabot matches our "commit hash with the tag in a comment" style.

❌ **Avoid:**

```yaml
- uses: actions/checkout@main  # Branches are regularly updated
- uses: actions/setup-python@v6.0.0  # Tags can be moved (even for releases)
```

### Pin action `runs-on:` labels to specific versions

Pin GitHub-hosted runner labels in
[`jobs.<job_id>.runs-on`](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idruns-on)
to specific versions from the
[available images list](https://github.com/actions/runner-images?tab=readme-ov-file#available-images)
for security and reproducibility.

Benefits:

- **Control:** Update runner versions on our schedule, not GitHub's
- **Reproducibility:** Consistent environment across time
- **Testing:** Can test changes before rolling out to all workflows

✅ **Preferred:**

```yaml
jobs:
  build:
    runs-on: ubuntu-24.04  # We can change this across our projects when we want
```

❌ **Avoid:**

```yaml
jobs:
  build:
    runs-on: ubuntu-latest  # This could change outside of our control
```

### Prefer Python scripts over inline Bash

Where possible, put workflow logic in Python scripts.

Benefits:

- **Testable:** Can be tested locally and with unit tests
- **Debuggable:** Easier to debug with standard Python tools
- **Portable:** Works consistently across platforms (Linux/Windows)
- **Approachable:** Better error handling and logging support
- **Modular:** Functions can be shared across multiple scripts

> [!TIP]
> Use your judgement for what logic is trivial enough to stay in bash.
>
> Some signs of complicated bash are _conditionals_, _loops_, _regex_,
> _piping command output_, and _string manipulation_.

✅ **Preferred:**

```yaml
- name: Process artifacts
  run: |
    python build_tools/process_artifacts.py \
      --families "${{ inputs.amdgpu_families }}" \
      --artifact-dir artifacts \
      --install-dir install
```

❌ **Avoid:**

```yaml
- name: Process artifacts
  shell: bash
  run: |
    for family in $(echo "${{ inputs.amdgpu_families }}" | tr ',' ' '); do
      if [[ -f "artifacts/${family}/rocm.tar.gz" ]]; then
        tar -xzf "artifacts/${family}/rocm.tar.gz" -C "install/${family}"
        echo "Extracted ${family}"
      else
        echo "::error::Missing artifact for ${family}"
        exit 1
      fi
    done
```

### Use safe defaults for inputs

Workflow inputs must have safe default values that work in common scenarios.

Benefits:

- **Safety:** Defaults don't trigger production changes
- **Fail-safe:** Mistakes default to non-destructive behavior
- **Developer-friendly:** Easy to use for common cases

> [!NOTE]
> Some workflows may be configured to have stricter security boundaries, such
> as only accepting "nightly" release types from certain branches or from
> certain repositories.

✅ **Preferred:**

```yaml
on:
  workflow_dispatch:
    inputs:
      release_type:
        type: choice
        description: Type of release to create. All developer-triggered jobs should use "dev"!
        options:
          - dev
          - nightly
          - prerelease
        default: dev  # Safe: development releases don't affect production

      amdgpu_families:
        type: string
        description: "GPU families to build (comma-separated). Leave empty for default set."
        default: ""  # Empty string handled gracefully in workflow logic
```

❌ **Avoid:**

```yaml
on:
  workflow_dispatch:
    inputs:
      release_type:
        type: choice
        description: "Type of release to create"
        options:
          - dev
          - nightly
          - stable
        default: nightly  # Unsafe: publishes to production
```

### Separate build and test stages

Use CPU runners to build from source and pass artifacts to test runners.

Benefits:

- **Cost optimization:** GPU runners are expensive; use them only when needed
- **Parallelization:** Multiple test jobs can share build artifacts
- **Packaging enforcement:** Testing in this way enforces that build artifacts
  are installable and usable on other machines

✅ **Preferred:**

```yaml
jobs:
  build_artifacts:
    name: Build Artifacts
    runs-on: azure-linux-scale-rocm  # Dedicated CPU runner pool for builds
    steps:
      # ...

      - name: Build ROCm artifacts
        run: |
          cmake -B build -GNinja .
          cmake --build build

      # ... Upload artifacts, logs, etc.

  test_artifacts:
    name: Test Artifacts
    needs: build_artifacts
    runs-on: linux-mi325-1gpu-ossci-rocm  # Expensive GPU runner only for tests
    steps:
      # ... Download artifacts, setup test environment, etc.

      - name: Run tests on GPU
        run: build_tools/github_actions/test_executable_scripts/test_hipblas.py
```

❌ **Avoid:**

```yaml
jobs:
  build_and_test:
    name: Build and Test
    runs-on: linux-mi325-1gpu-ossci-rocm  # Expensive GPU runner
    steps:
      # ...

      - name: Build ROCm artifacts
        run: |
          cmake -B build -GNinja .
          cmake --build build

      - name: Run tests on GPU
        run: build_tools/github_actions/test_executable_scripts/test_hipblas.py
```
