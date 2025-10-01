skip_tests = {
    "always": [
        # FLAKY!! AssertionError: 'tensor([2.3000+4.j, 7.0000+6.j])' != 'tensor([2.30000+4.j, 7.00000+6.j])'
        "test_print"
    ],
    "pytorch_version": {
        "2.10": [
            "test_terminate_handler_on_crash",  # hangs forever
        ]
    },
    "amdgpu_family": {},
}

# might be failing
# "test_index_add_correctness",
