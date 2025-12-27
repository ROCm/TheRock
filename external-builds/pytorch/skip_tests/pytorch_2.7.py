# PyTorch 2.7 specific skip tests
# Tests moved to generic.py have been removed to avoid duplication
skip_tests = {
    "common": {
        "cuda": [
            # This test was fixed in torch 2.9, see
            # https://github.com/ROCm/TheRock/issues/2206
            "test_hip_device_count",
        ]
    },
}
