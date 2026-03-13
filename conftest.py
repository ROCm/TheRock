def pytest_configure(config):
    config.addinivalue_line(
        "markers", 
        "amdsmi_tests_default_unblocking_for_sanity: marks tests as default-unblocking for amdsmi sanity (amdsmi_tests_default_unblocking_for_sanity)"
    )
    config.addinivalue_line(
        "markers", "amd_smi: marks tests that exercise the amd-smi CLI"
    )
