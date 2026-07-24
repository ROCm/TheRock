# JAX test selection (test sizes)

TheRock runs the JAX unit tests at three sizes, wired via the `test_size` input
of `.github/workflows/test_linux_jax_wheels_partial.yml`:

| Size     | What runs                                              | Where (trigger)              |
|----------|-------------------------------------------------------|------------------------------|
| `small`  | The curated subset in `small_tests.txt` (~40%)        | PR CI                        |
| `medium` | Full JAX UT suite + 2 end-to-end workloads            | Nightly (`release_type=nightly`) |
| `large`  | Full JAX UT suite + full end-to-end workloads         | Weekly (`release_type=prerelease`) |

Paths in `small_tests.txt` are relative to the checked-out `ROCm/jax` repo
(i.e. CI runs `pytest jax/<path>`).

## `small_tests.txt`

A static list of JAX unit tests selected to maximize LLM-weighted module
coverage on ROCm GPU. It is the smallest set that covers 100% of the weighted
module graph (coverage saturates at this size), i.e. the minimal
full-coverage set out of the 110-test ROCm candidate pool.

### How it was generated

Produced by the [`jax-test-selector`] tool (greedy weighted
max-coverage, Nemhauser–Wolsey–Fisher 1978), using the recommended `w+p`
configuration (`--alpha 1 --beta 1`):

```bash
cd jax-test-selector
PYTHONPATH=src python3 scripts/select_tests.py -k 55 \
  | grep -oE 'tests/[A-Za-z0-9_/]+\.py' | sort -u > small_tests.txt
```

`-k 55` is an upper bound; greedy stops early (42 tests) once every weighted
module is covered.

### Regenerating

Regenerate whenever JAX is bumped to a new major version.
