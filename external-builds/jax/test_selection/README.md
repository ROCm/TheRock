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
rule rather than taking a second slot in the pool. They are a placeholder that
warns and runs nothing until `ROCm/jax-lab` is public.

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
