# ROCm PR Quality — Evaluation Fixtures

Agent evaluation cases for the `rocm-pr-quality` base skill and the `therock-pr-quality` overlay. They exist so a change to either skill can be checked against fixed inputs and fixed expected findings, rather than by re-reading the prose. Deterministic helper behavior is covered separately by `tools/pr_quality_check_test.py`.

## How to use these

- Give the reviewing agent only the **Input** of a case (the PR description, the diff summary, and the evidence listed). Do not show it the **Expected** section.
- Grade the result against **Expected** and **Must not**. Grade a finding by what it says and its severity, not by exact wording.
- The inputs are inline fixtures, so they are pinned by construction and independent of any live repository. If you replay a case on a real PR, record its head SHA next to the result.
- The expected findings were written by hand, independently of any model run. Update them in the same PR as the skill change that moves them, and say why.
- These cases test judgment, not the helper. A model-based run is not part of CI; run them when changing the skills.

______________________________________________________________________

## Case 1: exempt docs-only change

**Input.** PR "Fix typo in build docs". Files: `docs/development/build_system.md` (+1/-1). Description: two sentences, no tracker, no waiver, no tests. CI green.

**Expected.**

- Overall assessment `APPROVED`.
- The change is classed `docs`; the self-evident exemption is applied (no tracker, no tests).
- No `BLOCKING` or `IMPORTANT` finding about tracking (M4) or tests (M2).

**Must not.** Ask the author to invent a waiver or a ticket. Treat a missing test as a gap.

______________________________________________________________________

## Case 2: product regression with an inadequate test

**Input.** PR "[FOO-101] Fix scan overflow for 64-bit offsets". Files: `src/scan.cpp` (+14/-3), `tests/scan_test.cpp` (+22). Class `defect-fix`. The new test asserts only that a mocked allocator was called; it never reads the scan output. A second test is a near-duplicate of the first with a different input size. CI green.

**Expected.**

- Overall assessment `CHANGES REQUESTED`.
- `BLOCKING`: the regression test passes against the unpatched code (M1; it asserts no behavior).
- `IMPORTANT`: no evidence that the new test fails before the fix.
- The near-duplicate test is at most a `SUGGESTION`.

**Must not.** Approve because a test file was added. Mark the near-duplicate `BLOCKING`.

______________________________________________________________________

## Case 3: compatible workflow change, caller unchanged (TheRock overlay)

**Input.** TheRock PR adds an optional input `verbose` (type boolean, default `false`) to a reusable `workflow_call` workflow. The one in-repo caller is not changed and does not pass the input. Behavior with the default is identical to before. Tracker present, CI green.

**Expected.**

- Overall assessment `APPROVED` (no `BLOCKING` finding).
- The review shows it inspected the callers and found the change compatible. It may note that external or version-pinned callers cannot be edited here and are unaffected by an optional input.

**Must not.** Require the caller to be edited. Mark the unchanged caller `BLOCKING`.

______________________________________________________________________

## Case 4: documented waiver is accepted

**Input.** PR "[FOO-202] Add experimental `fast_path` behind `FOO_FAST_PATH=0` default". Product code changed; no new tests; description declares `W-FLAG` with one sentence ("lands dark; soak tests tracked in FOO-203") and links FOO-203. CI green.

**Expected.**

- Overall assessment `APPROVED` with the waiver recorded, not blocked.
- The review adjudicates the waiver explicitly (accepts it and says why).

**Must not.** Mark M2 `BLOCKING` despite a valid, scoped waiver. Accept a waiver whose code does not address the rule (for example `W-NOTICKET` for missing tests).

______________________________________________________________________

## Case 5: incomplete evidence is stated, not papered over

**Input.** Same PR as Case 2 after the author fixes the test, but the reviewer cannot read the CI run (the logs have expired) and there is no local build.

**Expected.**

- The output has a limits note that names the reviewed commit and says CI evidence was unavailable and nothing was built or run.
- The review does not claim CI is green or that the test was seen to fail. A request to re-run CI or link a current run is an `IMPORTANT` finding.

**Must not.** State that the test passes or fails without evidence. Omit the limits note.

______________________________________________________________________

## Case 6: review, unsafe CI-label narrowing on a code-generation change

**Input.** Review of a rocm-libraries PR "[AIHPBLAS-1234] Reorder wait insertion in the gfx1250 scheduler". Files: `shared/stinkytofu/src/passes/WaitInsertion.cpp` (+40/-12) and its unit test. Labels: `ci:no-downstream-test`, `ci:platform:linux`. The description gives no reason for either label. The passing run used both labels. Tracker present, tests present. (StinkyTofu is linked into TensileLite's kernel generation for hipBLASLt and is built on Linux and Windows.)

**Expected.**

- Overall assessment `CHANGES REQUESTED`.
- `BLOCKING`: `ci:no-downstream-test` drops coverage the change can reach (code generation changes the kernels hipBLASLt ships, which rocBLAS and MIOpen consume).
- `BLOCKING` or `IMPORTANT` for `ci:platform:linux`: the code is compiled on Windows too.
- The review names the labels to remove and asks for a re-run.

**Must not.** Accept the labels because CI is green. Treat unsafe narrowing as a `SUGGESTION`.

______________________________________________________________________

## Case 7: review, more CI than the change needs

**Input.** Review of a TheRock PR "[ROCM-5678] Fix a typo in an error message in `build_tools/fetch_sources.py`" with a unit test update. Labels: `ci:run-all-archs`, `test_filter:comprehensive`. CI green. Tracker present.

**Expected.**

- Overall assessment `APPROVED`.
- A `SUGGESTION` that the labels run far more than the change can reach, naming the labels to drop.

**Must not.** Mark the extra CI `BLOCKING` or `IMPORTANT`. Withhold approval because of it.

______________________________________________________________________

## Case 8: pre-merge, CI not bypassed since review

**Input.** Pre-merge gate on an approved rocm-libraries PR. The review accepted the label set `ci:no-downstream-test` for a clients-only change, and `TheRock Multi-Arch CI` passed on commit `a1` with that set, building and testing hipBLASLt. After approval, the author pushed commit `b2`, which changes only `projects/hipblaslt/clients/README.md`, and added `ci:skip`. The `ci:skip` run on `b2` concluded `success` without building or testing. Timing and stale-base checks are clean.

**Expected.**

- Verdict go: the only change since the last tested pass is a CI-skippable docs file, so `ci:skip` bypassed nothing.

**Variant.** Same, but `b2` also changes `projects/hipblaslt/clients/common/include/testing_matmul.hpp`. **Expected:** verdict hold. Code changed after `ci:skip` was added, and the `b2` run's `success` built and tested nothing; recommend removing `ci:skip` and re-running CI.

**Must not.** Call the first case a hold because `ci:skip` is present. Treat the `ci:skip` run's `success` as coverage in the variant.
