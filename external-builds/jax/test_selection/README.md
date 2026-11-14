# JAX test selection

A selection file lists the tests to run instead of the whole JAX suite, one
path per line, relative to the `ROCm/jax` checkout. The `--test-list` option of
`run_jax_tests.py` applies it by ignoring everything under `tests/` the file
does not name, so the suite script still decides how the tests run:

```bash
python external-builds/jax/run_jax_tests.py --jax-dir jax \
  --test-list external-builds/jax/test_selection/small_tests.txt
```

Blank lines and `#` comments are ignored. A path the checkout does not have is
warned about and skipped, since test files come and go between JAX versions,
but a file naming nothing the checkout has fails the run rather than quietly
testing everything.

A selection is free to leave out the files the suite script names in its own
`--deselect` arguments: deselecting a node in a file that was never collected
is a no-op, not an error.

## Test sizes

CI picks a selection by test size. See `test_size` in
`.github/workflows/test_multi_arch_linux_jax_wheels.yml`.

| Size     | Single-GPU job                     | Multi-GPU job  | End-to-end workloads | Trigger                             |
| -------- | ---------------------------------- | -------------- | -------------------- | ----------------------------------- |
| `small`  | The subset in `small_tests.txt`    | No             | None                 | PR CI (`release_type=ci`)           |
| `medium` | The whole single-accelerator suite | One day a week | Two, on that day     | Nightly (`release_type=nightly`)    |
| `large`  | The whole single-accelerator suite | Yes            | All of them          | Release (`release_type=prerelease`) |

The multi-GPU job is a job of its own, on the family's
`test-runs-on-multi-gpu` runner, running `ci/run_pytest_rocm_multi.sh`. Sizes
decide whether it is worth a slot in that pool;
`build_tools/github_actions/configure_jax_test_matrix.py` holds the rule,
including which day of the week a `medium` run takes one.

The end-to-end workloads run as a step of that job, so they follow the same
rule rather than taking a second slot in the pool.

## End-to-end workloads

`e2e_medium.txt` and `e2e_large.txt` list the
[ROCm/jax-lab](https://github.com/ROCm/jax-lab) workloads each size runs, one
`<target> <workload> <gpus>` per line, with the GPU count jax-lab's own CI gives
each workload. `run_jax_e2e.py` runs them through jax-lab's
`targets/<target>/run.sh` against the wheels the job installed, and fails the
step if any of them fails:

```bash
python external-builds/jax/run_jax_e2e.py --jax-lab-dir jax-lab \
  --workload-list external-builds/jax/test_selection/e2e_medium.txt \
  --runner local --rocm-version 10.2.0a20261016
```

- They run on the Python 3.12 job only, the version jax-lab runs and pins.
- The test workflow checks out jax-lab at a pinned commit, so a jax-lab change
  reaches these runs through a PR here.
- Each workload gets 60 minutes, as in jax-lab's CI, and the step's budget is
  their sum.

`large` runs every jax-lab workload. `medium` runs two MaxText ones:
`llama3_1_8b`, the dense baseline, and `deepseek2_16b`, which adds
mixture-of-experts routing with shared experts, MLA attention and YaRN RoPE.
MaxText checks the most of any target (per-step TFLOP/s and step time on all 8
GPUs, against one final number for the others) and needs no downloaded data.
`mixtral_8x7b` and `gpt_oss_20b` peak at 69-92% of a 288 GB MI355X, more than
an MI300X's 192 GB, while these two stay under half.

## `small_tests.txt`

The smallest set of JAX tests that covers 100% of the LLM-weighted module
graph on ROCm GPU: greedy weighted max-coverage over the 110-test ROCm
candidate pool saturates at 42 files, plus `logging_test.py` for subprocess
stderr checks that catch stray rocprofiler-sdk output on XLA shutdown.

Regenerate it whenever JAX is bumped to a new major version, with the
`jax-test-selector` tool in its recommended `w+p` configuration:

```bash
cd jax-test-selector
PYTHONPATH=src python3 scripts/select_tests.py -k 55 \
  | grep -oE 'tests/[A-Za-z0-9_/]+\.py' | sort -u > small_tests.txt
```

`-k 55` is an upper bound; the greedy pass stops on its own once every weighted
module is covered.
