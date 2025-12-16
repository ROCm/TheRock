# Test Runner Setup

In the repo `TheRock`, `amdgpu_family_matrix.py` is the source of truth. We are allowed to do testing with new labels.

For other external repos that use TheRock CI, they get their GitHub self-hosted runner labels through the ROCm organization variable called `ROCM_THEROCK_TEST_RUNNERS`

We use the environment variable `LOAD_TEST_RUNNERS_FROM_VAR` with `amdgpu_family_matrix.py` to determine whether to pull from the `amdgpu_family_matrix.py` or `ROCm` organization variable

With this organization variable, we are able to update the runner labels immediately instead of having to open 2+ PRs.

TheRock CI gets this the runner labels via:

1. Retrieving the environment variable from ROCm organization (can be used in any repository in ROCm)
1. Parse the JSON string into Python dictionary
1. Adds the "test-runs-on" key / value in the associated amdgpu_family_info_matrix

The data for `ROCM_THEROCK_TEST_RUNNERS` is organized like so:

```
{
    "gfx110x": {
        "linux": "linux-gfx110X-gpu-rocm",
        "windows": "windows-gfx110X-gpu-rocm",
    },
    "gfx1151": {
        "linux": "linux-strix-halo-gpu-rocm",
        "windows": "windows-strix-halo-gpu-rocm",
    },
    "gfx90x": {
        "linux": "",
        "windows": "",
    },
    ...
}
```

For ROCm organization admin, please update the [runner labels in the ROCm organization settings](https://github.com/organizations/ROCm/settings/variables/actions/ROCM_THEROCK_TEST_RUNNERS)
