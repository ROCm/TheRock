def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "not_sanity: marks tests that must not run in sanity gating",
    )
    config.addinivalue_line(
        "markers",
        "amd_smi: marks tests that exercise amd-smi",
    )
    config.addinivalue_line(
        "markers",
        "amd_smi_cli: marks amd-smi CLI tests",
    )
