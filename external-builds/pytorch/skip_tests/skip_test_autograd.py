skip_tests = {
    "always": [],
    "pytorch_version": {
        "2.10": ["test_multi_grad_all_hooks", "test_side_stream_backward_overlap"]
    },
    "amdgpu_family": {},
}
