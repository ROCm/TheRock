# Reference: `trigger_coverage` job for the rocm-libraries nightly

This snippet belongs in **`ROCm/rocm-libraries`**
`.github/workflows/therock-multi-arch-ci-nightly.yml`, not in this repo. It is
kept here only as the canonical reference for the code-coverage dispatch
contract. TheRock cannot edit the external repo; copy this into the nightly when
wiring the real driver.

The nightly's regular build is **non-instrumented**, so it can only ever be the
baseline that coverage is measured against, never the thing coverage runs on.
The coverage orchestrator (`ROCm/TheRock`
`.github/workflows/multi_arch_ci_coverage.yml`) therefore has to run as a
**separate** GitHub Actions run with its own `run_id`; a nested `uses:` call
would share the nightly's `run_id` and collide the generic and instrumented
trees the installer needs to keep apart. The nightly hands over its own
`run_id` as `baseline_run_id`.

Cross-repo dispatch is authenticated with a **GitHub App token**
(`actions/create-github-app-token`), modelled on `ROCm/rocm-systems`
`.github/workflows/rocm_ci_caller.yml`. No classic PAT is introduced.

> **One-time org setup (not code):** a GitHub App reachable from rocm-libraries
> must be installed on **ROCm/TheRock** with **`actions: write`**, so the token
> minted below is allowed to call `createWorkflowDispatch` on TheRock.

## `workflow_dispatch` input (add to the nightly's `on:` block)

```yaml
on:
  workflow_dispatch:
    inputs:
      # ... existing inputs ...
      enable_coverage:
        description: >-
          Dispatch the ROCm/TheRock code-coverage orchestrator after the build
          and test jobs succeed. Nightly-only: a coverage run is a multi-hour
          instrumented build, unfit for presubmit.
        type: boolean
        default: false
```

## `trigger_coverage` job

```yaml
jobs:
  # ... existing setup, linux_build_and_test, etc. ...

  trigger_coverage:
    name: Trigger TheRock Coverage
    needs: [setup, linux_build_and_test]
    # Gate on the switch, on there being a build to baseline against, and on the
    # test job having succeeded -- coverage measures a run that actually built.
    if: >-
      ${{
        inputs.enable_coverage &&
        needs.setup.outputs.enable_build_jobs == 'true' &&
        needs.linux_build_and_test.result == 'success'
      }}
    runs-on: ubuntu-24.04
    steps:
      - name: Generate GitHub App token
        id: generate-token
        uses: actions/create-github-app-token@bcd2ba49218906704ab6c1aa796996da409d3eb1 # v3.2.0
        with:
          app-id: ${{ secrets.THEROCK_COVERAGE_APP_ID }}
          private-key: ${{ secrets.THEROCK_COVERAGE_APP_PRIVATE_KEY }}
          owner: "ROCm"
          repositories: "TheRock"

      - name: Dispatch TheRock coverage orchestrator
        uses: actions/github-script@f28e40c7f34bde8b3046d885e986cb6290c5673b # v8.0.0
        with:
          github-token: ${{ steps.generate-token.outputs.token }}
          script: |
            await github.rest.actions.createWorkflowDispatch({
              owner: 'ROCm',
              repo: 'TheRock',
              workflow_id: 'multi_arch_ci_coverage.yml',
              // Pin to a known-good TheRock ref; bump when the coverage
              // workflows change in a way this nightly depends on.
              ref: 'main',
              inputs: {
                // The nightly's own run supplies the non-instrumented baseline.
                baseline_run_id: String(context.runId),
                baseline_repository: 'ROCm/rocm-libraries',
                // Match the channel this nightly publishes to.
                baseline_release_type: 'nightly',
                // Records the exact rocm-libraries commit that drove coverage.
                coverage_config_source: '${{ github.repository }}@${{ github.sha }}',
                projects_to_test: 'hipblas,hipdnn,hipfft,hiprand,hipsolver,hipsparselt,rocblas,rocfft,rocrand,rocsolver,rocsparse',
                amdgpu_families: 'gfx94X-dcgpu',
              },
            });
```

Notes:

- `baseline_run_id` is stringified because `createWorkflowDispatch` inputs must
  be strings; `context.runId` is a number.
- `ref` pins which TheRock copy of the coverage workflows runs. A
  `workflow_dispatch` always executes the workflow file from that ref.
- `baseline_release_type` and `projects_to_test` should track whatever the
  nightly actually built and published; the values above match the phase 1
  default selection.
