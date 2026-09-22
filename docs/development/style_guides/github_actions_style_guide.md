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
  env:
    AMDGPU_FAMILIES: ${{ inputs.amdgpu_families }}
  run: |
    python build_tools/process_artifacts.py \
      --families "${AMDGPU_FAMILIES}" \
      --artifact-dir artifacts \
      --install-dir install
```

❌ **Avoid:**

```yaml
- name: Process artifacts
  shell: bash
  env:
    AMDGPU_FAMILIES: ${{ inputs.amdgpu_families }}
  run: |
    for family in $(echo "${AMDGPU_FAMILIES}" | tr ',' ' '); do
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
> Release workflows in TheRock should only offer "dev" for manual dispatch.
> CI-oriented build/test helpers may offer "ci" and "dev"; rockrel-owned
> callers can pass "nightly" and "prerelease" through `workflow_call`.

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
    runs-on: aws-linux-scale-rocm-prod  # Dedicated CPU runner pool for builds
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
    runs-on: linux-gfx942-1gpu-ossci-rocm  # Expensive GPU runner only for tests
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
    runs-on: linux-gfx942-1gpu-ossci-rocm  # Expensive GPU runner
    steps:
      # ...

      - name: Build ROCm artifacts
        run: |
          cmake -B build -GNinja .
          cmake --build build

      - name: Run tests on GPU
        run: build_tools/github_actions/test_executable_scripts/test_hipblas.py
```

### Security guidelines

Use [zizmor](https://docs.zizmor.sh/) to check workflows and composite actions.
See its [audit reference](https://docs.zizmor.sh/audits/) for rule details.

```bash
# See https://docs.zizmor.sh/installation/ for other options
pip install zizmor

# Check workflows and composite actions locally.
zizmor --offline .github

# Recheck a workflow after editing it.
zizmor --offline .github/workflows/multi_arch_ci.yml
```

Confirm the findings you addressed no longer appear; unrelated findings may
still produce a nonzero exit code. `--offline` skips audits requiring network
access. Add `--no-ignores` to inspect suppressed findings too.

#### Security - Avoid template injection

Pass input values through step environment variables instead of inserting
`${{ ... }}` expressions into `run:`. GitHub expands expressions before the
shell parses the script, so even a quoted expression can execute injected code.
See [template-injection](https://docs.zizmor.sh/audits/#template-injection).

✅ **Preferred:**

```yaml
# Here, a value such as `$(cat file.txt)` is passed literally to Python.
- name: Process artifacts
  shell: bash
  env:
    AMDGPU_FAMILIES: ${{ inputs.amdgpu_families }}
  run: |
    python build_tools/process_artifacts.py --families "${AMDGPU_FAMILIES}"
```

❌ **Avoid:**

```yaml
# Here, a value such as `$(cat file.txt)` is executed by the shell (!).
- name: Process artifacts
  run: |
    python build_tools/process_artifacts.py --families "${{ inputs.amdgpu_families }}"
```

**Exceptions:** A value intentionally supplying shell-quoted arguments or a
script may require direct expansion. Only allow this when its source is trusted
to supply executable code. Document that assumption beside a local suppression:

```yaml
- name: Configure
  # The caller is trusted to supply executable build options, including shell syntax.
  run: | # zizmor: ignore[template-injection]
    cmake -B build ${{ inputs.cmake_args }}
```

#### Security - Limit usage and forwarding of secrets

Minimize reliance on secrets for two reasons:

- **Contributor compatibility:** `pull_request` runs from forks do not normally
  receive repository or organization secrets. Keep core build and test paths
  working without secrets so both internal _and external_ contributors can use
  workflows.
- **Security:** Give workflows only the credentials and permissions they need
  to reduce the attack surface and impact of compromised code.

To address zizmor's [secrets-inherit](https://docs.zizmor.sh/audits/#secrets-inherit)
findings:

- **If no secrets are needed:** Omit `secrets:` lines and add a short comment
  highlighting it, as below.
- **If specific secrets are needed:** Forward them by name and declare them in the
  callee's `on.workflow_call.secrets`.
- **Intentional inheritance:** Higher-trust entry points may use
  `secrets: inherit` to avoid repeating secret mappings across nested workflows.
  Child workflows may also inherit a limited set of secrets explicitly forwarded
  by a limited-trust entry point. Add a short comment explaining the trust level
  or caller-imposed limits and suppress the finding with
  `# zizmor: ignore[secrets-inherit]`.

✅ **Preferred:**

When the called workflow needs no secrets, omit forwarding and document why.

```yaml
on:
  pull_request:

jobs:
  test:
    uses: ./.github/workflows/test.yml
    # Note: not using 'secrets: inherit' here; no secrets are needed.
```

✅ **Preferred:**

When a workflow needs secrets, explicitly forward them and declare them in the
callee's `on.workflow_call.secrets`.
Across a workflow chain, each call must repeat the mappings and each callee must
repeat the declarations.

> [!TIP]
> Using `secrets: inherit` can avoid this duplication in higher-trust workflows
> or when a limited-trust entry point restricts the forwarded set (see below).

```yaml
# .github/workflows/release.yml (caller)
on:
  workflow_dispatch:

jobs:
  notify:
    uses: ./.github/workflows/notify.yml
    secrets:
      GH_APP_HAULY_CID: ${{ secrets.GH_APP_HAULY_CID }}
      GH_APP_HAULY_PRIVATE_KEY: ${{ secrets.GH_APP_HAULY_PRIVATE_KEY }}
```

```yaml
# .github/workflows/notify.yml (callee)
on:
  workflow_call:
    secrets:
      GH_APP_HAULY_CID:
        required: true
      GH_APP_HAULY_PRIVATE_KEY:
        required: true

jobs:
  notify:
    runs-on: ubuntu-24.04
    steps:
      - name: Notify Quartz
        uses: ROCm/Quartz/.github/actions/notify_quartz@f386a9756620938616af0b4d5d04b24ae6e0353f # notify_quartz/v1.1.1
        with:
          gh_app_client_id: ${{ secrets.GH_APP_HAULY_CID }}
          gh_app_private_key: ${{ secrets.GH_APP_HAULY_PRIVATE_KEY }}
          run_phase: started
          reporting_workflow: notify.yml
```

🟡 **Acceptable with tradeoffs:**

Higher-trust entry points may inherit secrets to avoid repeating mappings
across nested workflows.

```yaml
on:
  push:
    branches: [main]
  schedule:
    - cron: "0 0 * * *"
  workflow_dispatch:

jobs:
  build:
    # Higher-trust entry points: triggered by maintainers or automation.
    uses: ./.github/workflows/build.yml
    secrets: inherit # zizmor: ignore[secrets-inherit]
```

Child workflows may also inherit secrets after a limited-trust entry point
explicitly restricts the forwarded set. Here, nested calls receive only the two
named credentials. Fork PRs still lack these secrets, so secret-dependent
notifications must handle their absence.

```yaml
# .github/workflows/ci.yml
on:
  pull_request:

jobs:
  test:
    uses: ./.github/workflows/test.yml
    secrets:
      GH_APP_HAULY_CID: ${{ secrets.GH_APP_HAULY_CID }}
      GH_APP_HAULY_PRIVATE_KEY: ${{ secrets.GH_APP_HAULY_PRIVATE_KEY }}
```

```yaml
# .github/workflows/test.yml
on:
  workflow_call:
    secrets:
      # The PR entry point explicitly limits the forwarded set.
      GH_APP_HAULY_CID:
        required: false
      GH_APP_HAULY_PRIVATE_KEY:
        required: false

jobs:
  component:
    uses: ./.github/workflows/test_component.yml
    secrets: inherit # zizmor: ignore[secrets-inherit]
```

❌ **Avoid:**

Do not forward all available secrets to a workflow that does not need them.

```yaml
on:
  pull_request:

jobs:
  test:
    uses: ./.github/workflows/test.yml
    secrets: inherit
```
