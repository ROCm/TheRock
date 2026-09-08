# JAX test skip lists

Tests that are known to fail for reasons outside JAX live here rather than in a
workflow file, so that a local run reproduces what CI runs.

## Layout

| File               | Applies to                                  |
| ------------------ | ------------------------------------------- |
| `generic.py`       | every JAX version                           |
| `jax_<version>.py` | that JAX version only, e.g. `jax_0.10.2.py` |

Each file defines a `skip_tests` dict whose top-level keys select where the
entries apply:

- `common` applies everywhere.
- Anything else is matched against the GPU family, as a substring: `gfx94`
  covers the family `gfx94X-dcgpu` and the arch `gfx942`.

## Entry format

Entries are pytest `-k` keyword filters. `deny` is a keyword to skip, and the
optional `unless` list keeps tests that would otherwise be caught by it, which
matters because `-k` matches substrings: denying `conv` also catches `convert`.

```python
skip_tests = {
    "gfx94": {
        "keywords": [
            {"deny": "conv", "unless": ["convert", "conversion"]},
            {"deny": "sumpool"},
        ],
    },
}
```

That generates:

```
-k "((not conv) or convert or conversion) and not sumpool"
```

Always say why an entry is there and what would let it be removed. An entry
without a reason cannot be retired by the next person to read it.

## Inspecting and running locally

Print the expression a given configuration produces:

```bash
python external-builds/jax/skip_tests/create_skip_tests.py \
    --jax-version 0.10.2 --amdgpu-family gfx94X-dcgpu
```

`run_jax_tests.py` applies the same expression, so running it locally with the
same `--jax-version` and `--amdgpu-family` skips the same tests as CI:

```bash
python external-builds/jax/run_jax_tests.py --jax-dir jax \
    --jax-version 0.10.2 --amdgpu-family gfx94X-dcgpu
```

To debug one of these tests, invert the list and run only the skipped tests:

```bash
python external-builds/jax/run_jax_tests.py --jax-dir jax \
    --jax-version 0.10.2 --amdgpu-family gfx94X-dcgpu --debug
```
