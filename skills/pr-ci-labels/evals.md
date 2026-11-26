# PR CI labels — Evaluation Cases

Agent evaluation cases for the `pr-ci-labels` skill, so a change to the skill can be checked against fixed inputs and expected answers instead of by re-reading the prose. The `ci_labels.py` report's behavior is covered separately by `tools/ci_labels_test.py`.

## How to use these

- Give the agent only a case's **Input** (and, for real PRs, the PR and its pinned head SHA). Do not show it **Expected** or **Must not**.
- Grade by what the answer says, not its exact wording.
- Part A uses inline fixtures, so it is pinned by construction. Part B uses real PRs: read the diff at the pinned head SHA (for example `gh api repos/<owner>/<repo>/commits/<sha>` per commit, or the PR's file list if the head has not moved), and record the SHA you used next to the result.
- Expected answers were written by hand. Update them in the same PR as the skill change that moves them, and say why.
- Record the model, the wall time, and the estimated cost for each case, not only whether the answer was right. The skill is meant to answer in about 10 to 30 seconds and cheaply enough to run on every PR eventually. A first run on November 18, 2026 took 20 to 104 seconds and an estimated 0.6 to 5.8 US cents per case, depending on the model.

______________________________________________________________________

## Part A: fixtures

### Case A1: suggest, docs-only change

**Input.** Suggest labels for a rocm-libraries branch that changes only `projects/rocsparse/README.md` (+12/-3) and `docs/install.rst` (+4/-1). The repository's `.github/skip-ci-config.toml` lists `*.md`, `*.rst`, and `docs/*`.

**Expected.** No labels: every changed path is auto-skipped, so CI does not build or test. The description line says "no CI labels needed: CI skips these paths."

**Must not.** Propose `ci:skip` as if it were needed, or add GPU-family or `test_filter:` labels.

### Case A2: suggest, clients-only change

**Input.** Suggest labels for a rocm-libraries branch whose four changed files are all under `projects/hipblaslt/clients/` (the `hipblaslt-test` and `hipblaslt-bench` CPU-verification code). No library, header, kernel, or TensileLite file changes.

**Expected.**

- `ci:no-downstream-test`, with the reason that consumers cannot see client code.
- Says what it drops (the rocBLAS, MIOpen, hipBLAS, and hipSPARSELt test jobs; hipBLASLt-provider is also acceptable).
- Drafts the `gh pr edit --add-label` command without running it, and says to apply it before the PR is marked ready.

**Must not.** Propose `ci:skip` or `ci:cpu-test-only` (the client code runs inside GPU tests). Apply the label itself.

### Case A3: check, unsafe narrowing on a code-generation change

**Input.** Check the labels on a rocm-libraries PR that changes `shared/stinkytofu/src/passes/WaitInsertion.cpp` (+40/-12) and its unit test. Labels: `ci:no-downstream-test`, `ci:platform:linux`. (StinkyTofu is linked into TensileLite's kernel generation for hipBLASLt and is built on Linux and Windows.)

**Expected.**

- `ci:no-downstream-test` drops coverage the diff can reach: code generation changes the kernels hipBLASLt ships, which rocBLAS and MIOpen consume.
- `ci:platform:linux` drops the Windows build of code compiled on both platforms.
- Suggests removing both, as drafted commands.

**Must not.** Call either label safe because CI is green.

### Case A4: check, more CI than needed

**Input.** Check the labels on a TheRock PR that fixes a typo in an error message in `build_tools/fetch_sources.py` and updates its unit test. Labels: `ci:run-all-archs`, `test_filter:comprehensive`.

**Expected.** Both labels run more than the diff needs: a host-side Python string change cannot change defaults, supported devices, or anything only the comprehensive tier tests. Suggests dropping both.

**Must not.** Say the labels drop coverage, or treat the extra CI as unsafe.

______________________________________________________________________

## Part B: real PRs

Suggest labels for each PR at its pinned head SHA, then compare with the expected set.

| PR                                                                        | Head SHA     | What it changes                                                                    | Expected labels                                                                                                            | Must not                                                                |
| ------------------------------------------------------------------------- | ------------ | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [rocm-libraries#15973](https://github.com/ROCm/rocm-libraries/pull/15973) | `59d0ff8746` | 4 files under `projects/hipblaslt/clients/`                                        | `ci:no-downstream-test` (`ci:asan` acceptable: it changes host-buffer lifetime)                                            | `ci:skip`, `ci:cpu-test-only`                                           |
| [rocm-libraries#15633](https://github.com/ROCm/rocm-libraries/pull/15633) | `145310ee72` | 6 files under `shared/stinkytofu/` (gfx1250 scheduler)                             | No narrowing labels (default CI); gfx125X build and test labels are acceptable                                             | `ci:no-downstream-test`, `ci:platform:*`, `ci:skip`, `ci:cpu-test-only` |
| [TheRock#6653](https://github.com/ROCm/TheRock/pull/6653)                 | `c46dc78428` | 7 files under `skills/` (it carried `ci:skip`)                                     | None: `skills/*` is auto-skipped, so `ci:skip` added nothing                                                               | Saying `ci:skip` was needed                                             |
| [TheRock#10077](https://github.com/ROCm/TheRock/pull/10077)               | `40057176a7` | The `rocm-libraries` submodule pointer (a bot bump)                                | Keep `ci:run-all-archs` and `ci:asan` (the bump convention)                                                                | Any narrowing label                                                     |
| [TheRock#10025](https://github.com/ROCm/TheRock/pull/10025)               | `590e462917` | One file under `skills/`                                                           | None: auto-skipped                                                                                                         | Any label                                                               |
| [rocm-libraries#15636](https://github.com/ROCm/rocm-libraries/pull/15636) | `7baaf2a052` | 3 files under `projects/hipblaslt/clients/`                                        | `ci:no-downstream-test`                                                                                                    | `ci:skip`, `ci:cpu-test-only`                                           |
| [rocm-libraries#16167](https://github.com/ROCm/rocm-libraries/pull/16167) | `d9fa6b9753` | hipTensor plan-cache double free and LRU eviction fix, with a test                 | `ci:asan`                                                                                                                  | Omitting `ci:asan`; `ci:skip`                                           |
| [rocm-libraries#15989](https://github.com/ROCm/rocm-libraries/pull/15989) | `49fce65825` | 4 files under `shared/stinkytofu/` (gfx1250 instruction tables and the asm bridge) | No narrowing labels; a gfx125X label with `ci:test:gfx125x` to test gfx1250 is acceptable                                  | Any narrowing label; misstating what default CI builds                  |
| [TheRock#10061](https://github.com/ROCm/TheRock/pull/10061)               | `f5dfe81501` | 3 build workflows (Linux, Windows, WSL); the PR is a draft                         | No narrowing labels; says CI will not run while it is a draft; `ci:run-wsl` only with the WSL workflow named as the reason | Suggesting `ci:run-on-draft` as coverage; opt-in labels with no reason  |
