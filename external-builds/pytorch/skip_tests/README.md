# Skipping pytorch tests

## Introduction

This tooling allows to narrow down pytorch tests by skipping explicitely tests.
Either in general, or additioanlly filtered by `amdgpu_family` and/or `pytorch_version`.

By default, we are trying to follow the recommended [PyTorch test suite](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html#testing-the-pytorch-installation).
Below you find the extract of it (1st Oct, 2025).

```
PYTORCH_TEST_WITH_ROCM=1 python3 test/run_test.py --verbose \
--include test_nn test_torch test_cuda test_ops \
test_unary_ufuncs test_binary_ufuncs test_autograd
```

Note:
1st Oct 2025: Missing is `test/test_ops.py` and `test/inductor/test_torchinductor.py` due to issues in Triton.

However, the filtering using `PYTORCH_TEST_WITH_ROCM=1` is not fully reflecting our failures. As such, this tooling will provide a more fine-grained filtering mechanism in our control to skip additional tests.

Independent of this tooling, it is _always_ welcome to _get those changes upstream_! :)

## How to run

[`../run_pytorch_tests.py`](../run_pytorch_tests.py) steers the pytest and is
used by the CI, while [`./create_skip_tests.py`](create_skip_tests.py) creates
the list of tests to be included or excluded.

## Structure

The files including the tests to be skipped are:

- `generic.py` is always included to skip tests
- `pytorch_<version>.py` is only included for the given PyTorch version to skip tests

Within those files, the following structure is used

```py
skip_tests = {
    "common": {
        <PyTorch test module> : [ <Tests> ],
    },
    "<amdgpu family>": {,
        <PyTorch test module> : [ <Tests> ],
    },
}
```

The PyTorch test modules are `nn`, `cuda`, `unary_ufuncs` etc.
This ordering is mainly added for easier debugging as otherwise it is difficult to determine which test module contains a given tests like `test_host_memory_stats` belongs to.

When adding new tests to be skipped, consider adding a small comment why it was added, and best if there is any condition/resolution waiting when it can be taken off again.

An example how it could look like is given below:

```py
skip_tests = {
    "common": {
        "cuda": [
            "test_device_count_not_cached_pre_init",
            "test_host_memory_stats",
        ]
    },
    "gfx942": {
        "autograd": [
            "test_multi_grad_all_hooks",
            "test_side_stream_backward_overlap"
            ],
        "cuda": [
            "test_cpp_memory_snapshot_pickle",
            "test_memory_compile_regions",
        ],
        "nn": [
            "test_side_stream_backward_overlap"
        ],
        "torch": [
            "test_terminate_handler_on_crash",  # hangs forever
        ]
    },
}
```

## How to: Upstream skipped tests to PyTorch

For example:
Error message

```
FAILED [0.2901s] external-builds/pytorch/pytorch/test/test_cuda.py::TestCudaMallocAsync::test_memory_compile_regions - TypeError: 'CustomDecompTable' object is not a mapping
```

1. Go to [GitHub PyTorch](https://github.com/pytorch/pytorch)
1. Search for the class name `TestCudaMallocAsync`
1. Find the function `test_memory_compile_regions`
1. Decide further steps, e.g add `@skipIfRocm`

Function description of @skip\<..> can be found in `torch/testing/_internal/common_utils.py`.
They include

```
@skipIfRocm
@skipIfRocmArch(MI300_ARCH)
@unittest.skipIf(not has_triton(), "test needs triton")
...
```
