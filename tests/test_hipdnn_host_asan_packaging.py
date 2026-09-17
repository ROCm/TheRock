# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_hipdnn_integration_host_asan_runtime_closure_is_packaged():
    descriptor = _load_toml(
        REPO_ROOT / "ml-libs" / "artifact-hipdnn-integration-tests.toml"
    )
    test_component = descriptor["components"]["test"][
        "ml-libs/hipdnn_integration_tests/stage"
    ]["include"]
    assert "bin/*hipdnn_*_test*" in test_component
    assert "bin/hipdnn_integration_tests_ctest/CTestTestfile.cmake" in test_component
    assert "lib/integration-test-bundles/**" in test_component

    topology = _load_toml(REPO_ROOT / "BUILD_TOPOLOGY.toml")
    dependencies = topology["artifacts"]["hipdnn-integration-tests"]["artifact_deps"]
    assert "rand" in dependencies


def test_hipdnn_synthetic_engine_plugin_runtime_closure_is_packaged():
    descriptor = _load_toml(REPO_ROOT / "ml-libs" / "artifact-hipdnn.toml")
    stage = "ml-libs/hipDNN/stage"
    assert "bin/hipdnn_list_engines*" in descriptor["components"]["run"][stage][
        "include"
    ]
    assert "lib/test_plugins/**" in descriptor["components"]["test"][stage][
        "include"
    ]
