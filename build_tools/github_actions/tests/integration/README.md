# CI configuration integration tests

End-to-end scenarios for `configure_multi_arch_ci.py`. A scenario describes a
GitHub event and a git diff, runs the **real** pipeline, and asserts on the
`GITHUB_OUTPUT` a workflow would consume.

Nothing inside the pipeline is mocked. The unit tests in
`../configure_multi_arch_ci_test.py` cover individual functions; these cover
how those functions compose, which is where behavior has historically drifted.

```
scenarios/pipeline/*.yaml   the scenarios
scenario_schema.py          YAML format + validation
scenario_runner.py          executes a scenario, asserts expectations
mock_family_matrix.py       frozen GPU family matrix
conftest.py                 hermeticity guards
guards_test.py              tests that the guards actually fire
```

## Adding a scenario

Append to the relevant file in `scenarios/pipeline/`:

```yaml
- name: ci-skip label skips the entire pipeline
  description: >
    Optional. Say *why* this behavior matters, not what the YAML already says.
  trigger:
    event: pull_request        # push | schedule | workflow_dispatch
    ref: my-feature            # GITHUB_REF_NAME, defaults to "main"
    payload:                   # the GitHub event payload, verbatim
      pull_request:
        base: {ref: main}
        labels: [{name: "ci:skip"}]
  env:                         # optional, layered over conftest.BASE_ENV
    BUILD_VARIANT: release
  git:                         # omit entirely for an empty GitContext
    changed_files: ["core/CMakeLists.txt"]
    submodule_paths: []
  expect:
    outputs:                   # raw GITHUB_OUTPUT values, as strings
      enable_build_jobs: "false"
    linux:                     # or `linux: null` to assert the platform skipped
      families: ["mock-primary"]
      tested_families: []
      build_pytorch: true
    jobs:
      build_rocm: "run"
```

Expectations are **partial** — only the keys you name are checked. Adding a
field to `BuildConfig` won't invalidate existing scenarios, but a scenario that
cares about it can opt in by naming it.

Unknown keys are rejected at collection time. A silently ignored typo in an
`expect` block is a test that passes while asserting nothing.

### Available expectations

| Block | Meaning |
|---|---|
| `outputs.<key>` | Exact match against the parsed `GITHUB_OUTPUT` file |
| `linux` / `windows` | Fields of the platform's build config, or `null` to assert it was skipped |
| `jobs.<group>` | `run`, `prebuilt`, or `skip`. Also `test_type_reason`. |
| `summary_contains` | Substrings that must appear in the step summary |
| `expect_error` | Substring of the exception a scenario must raise (replaces `expect`) |

Inside `linux` / `windows`, three keys are derived conveniences:
`families`, `tested_families`, and `wsl_families`. Everything else is matched
directly against the build config dict, so it uses the serialized shape the
workflow sees (`prebuilt_stages` is a comma-joined string, not a list).

## GPU families

Scenarios default to `families: mock`, a frozen matrix in
`mock_family_matrix.py`. The real matrix changes whenever a gfx target moves
between runner pools, and scenarios asserting on real names would break on
every one of those edits for reasons unrelated to the behavior under test.

Set `families: real` when a scenario genuinely needs production names. Stage
reuse requires this — it resolves artifact names through
`expand_amdgpu_families`, which rejects mock family names.

Invariants over the real matrix live in `family_matrix_contract_test.py` and
are written as properties (`tests_on_trigger ⊆ builds_on_trigger`) rather than
golden values, so they don't churn either.

## Hermeticity

These scenarios are only meaningful if the pipeline can't reach outside the
test. Four autouse guards in `conftest.py` enforce that, and `guards_test.py`
verifies each one still fires:

- **Environment** is wiped to `BASE_ENV`. `amdgpu_family_matrix` reads
  `CI_CONFIG_PATH`; `stage_reuse_decision` reads seven `STAGE_REUSE_*`
  variables. Without this, results depend on the developer's shell.
- **Network** is blocked on *both* GitHub transports — `urlopen` and the
  `gh api` subprocess — and the cached auth method is pinned. Note
  `STAGE_REUSE_MODE` defaults to `dry-run`, not `off`, so the baseline lookup
  is live unless a scenario disables it; `BASE_ENV` sets it to `off`.
- **Git** subprocesses are blocked. Scenarios supply their diff via `git:`.
- **Runner selection** is pinned. `select_weighted_label` uses
  `random.SystemRandom().choices()`, so `build_runs_on_*` would otherwise flake.

The TOML files (`BUILD_TOPOLOGY.toml`, `skip-ci-*.toml`) are deliberately
**not** stubbed. Reading the real ones is the point: a topology change that
breaks job decisions should fail these tests.

## Running

```bash
pytest build_tools/github_actions/tests/integration/

# One scenario, by name
pytest build_tools/github_actions/tests/integration/ -k "ci-skip label"
```

A mismatch reports every failing field at once:

```
AssertionError: Scenario 'pull_request::submodule bump' did not match:
  outputs.test_type:
      expected: 'full'
      actual:   'standard'
```
