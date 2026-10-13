# Native Windows DVC-cache comparison for TheRock #8807 / #8505

Prepared October 13, 2026 by toga-6-shark52. This is a local validation artifact, outside the PR. No workflow was published or dispatched.

The bundle contains verbatim fetcher snapshots from current main (`baseline`, b04778ebc8c987fec84b902ec80fcb5a9974c4f2), rebased PR #8807 (`candidate`, b4236cc48911cd7250a4786a1e91778fb79a79f7), and the TheRock revision used by the October 12 failure (`incident-baseline`, 015ba176c2db9bfaf5f9a2e7a774bd3e88b72a74). Each snapshot has a revision.txt; results also record the source SHA256.

Three exact input subsets are available:

- `oct12-batchnorm`: 12 Batchnorm pointers / 72 outputs / 496,320 bytes from libraries 1d5f09453d612234e6e55318d8a137c46a053f98, the external checkout recorded in [#13286's failed October 12 job](https://github.com/ROCm/rocm-libraries/actions/runs/38306051918/job/115001235090). Includes failing hash cee73630e22266547f7e77a86560f21b, referenced by six outputs.
- `oct06-batchnorm`: the same size workload from libraries develop 716a1275de5da9b49198a61934f939daa97cb8d8, [October 6 job](https://github.com/ROCm/rocm-libraries/actions/runs/37497006510/job/112399948948).
- `oct06-sdpa`: three pointers / 21 outputs / 16,515,180 bytes sharing b8852f60c69d1ec2ae2bc2218511246f, from #12737 head cb2d09dad2e98093a17106195f18f89f09d43ee0, [October 6 job](https://github.com/ROCm/rocm-libraries/actions/runs/37482550001/job/112355355098).

`fixture-provenance.json` records repository revisions, paths and SHA256 digests. No binary fixture data is bundled; the fetcher downloads real public S3 objects.

## Run on native Windows

Copy this directory to a native Windows machine, or temporarily publish it with the draft workflow in `workflow-dispatch.yml`. No ROCm build or GPU is needed. WSL does not exercise Windows rename semantics.

From PowerShell, in this directory:

```powershell
py -3.12 -m venv .\venv
.\venv\Scripts\python.exe -m pip install "boto3>=1.42.79,<1.42.85"
.\venv\Scripts\python.exe replay.py --output .\results-oct12 --fixture oct12-batchnorm --iterations 100 --jobs 12
.\venv\Scripts\python.exe replay.py --output .\results-sdpa --fixture oct06-sdpa --iterations 100 --jobs 12
```

Output paths must be new. Each round starts a separate process for each fetcher, alternates execution order, creates a new cache and new destinations, runs the normal pull with 12 pointer workers, then tests a warm-cache pull into new destinations. Success requires all expected outputs, verified size/MD5 for destinations and cache files, and no leftover staging files. Worker exceptions are logged and rethrown without changing the fetch operations. The controller records failures and continues to the next round. It returns nonzero if any round fails.

For the historical fetcher versus the candidate, add `--variants incident-baseline candidate`. For Windows CI-like filesystem behavior, repeat on `aws-windows-scale-rocm-prod-mix`, the pool used by the failed jobs. The supplied workflow defaults to hosted `windows-2022`; changing its runner input to the AWS pool permits a closer environment match if authorized.

Keep the top-level results.json plus per-round pull.log, process.log and result.json. A baseline failure in cache publication/hashing followed by candidate success is stronger evidence than candidate success alone. If both variants pass, the run establishes successful pulls but does not reproduce #8505. If the candidate fails, the traceback identifies which cache operation still needs attention.

100 zero-failure rounds give an approximate 95% upper bound of 3% per round under an independent, identical-trial model. Real CI jobs differ in cache/runner state and contention, so this is not a fleet-wide error-rate bound. Increase repetitions only if the baseline fails often enough to make an A/B comparison useful. A native deterministic interleaving test may be needed if spontaneous reproduction is too rare.

The small subsets preserve implicated duplicate hashes; they do not recreate the original job's complete 57-pointer download load or runner state. If the subset does not reproduce the issue, expand to the complete original DVC pointer set with both variants and keep the same worker count, then test on the affected runner pool. Do not close #8505 based on a single green full build.

## Why rerunning the old GitHub run is insufficient

GitHub reruns reuse the original run's commit and workflow dependencies. #12737's workflow pins TheRock 498444ae814a961ace81aed8717054208e41a7d8; #13286's failed run uses 015ba176c2db9bfaf5f9a2e7a774bd3e88b72a74. Neither includes #8807. A replay must explicitly select the candidate fetcher, or a temporary libraries integration branch must update every relevant reusable-workflow `uses: ...@revision` and TheRock checkout `ref:` together. Keep that test branch independent of unrelated PR changes.

The draft workflow assumes this bundle is committed under `validation/dvc-cache-replay/` and the workflow file is placed under `.github/workflows/`. Publishing the files and dispatching the workflow are separate network writes; neither has been performed.

## Prepared validation branch

Branch `shared/dvc-cache-windows-validation-20261013` replaces `.github/workflows/unit_tests.yml` only on this temporary branch with a manual-only two-job Windows comparison. It uses the workflow ID already present on main, permitting dispatch without adding a workflow to the default branch. Each job runs 100 baseline and 100 candidate rounds on windows-2022; jobs cover oct12-batchnorm and oct06-sdpa. Timeout: 30 minutes per job. No ROCm build, GPU job, or automatic PR trigger. Results are uploaded even when a replay fails.
