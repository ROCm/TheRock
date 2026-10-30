# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end scenarios for the multi-arch CI configuration pipeline.

Each scenario in ``scenarios/pipeline/`` runs the real pipeline - no internal
function is mocked - from a GitHub event payload through to the
``GITHUB_OUTPUT`` a workflow would consume. See ``scenario_schema.py`` for the
YAML format and ``README.md`` for how to add one.
"""

import pytest
from scenario_runner import assert_scenario, run_scenario
from scenario_schema import load_scenarios

PIPELINE_SCENARIOS = load_scenarios("pipeline")


@pytest.mark.parametrize(
    "scenario",
    PIPELINE_SCENARIOS,
    ids=[scenario.id for scenario in PIPELINE_SCENARIOS],
)
def test_pipeline_scenario(scenario, tmp_path):
    if scenario.expect_error:
        with pytest.raises(Exception) as excinfo:
            run_scenario(scenario, tmp_path)
        assert scenario.expect_error in str(excinfo.value), (
            f"Scenario {scenario.id!r} raised {excinfo.value!r}, "
            f"which does not contain the expected text {scenario.expect_error!r}"
        )
        return

    assert_scenario(scenario, run_scenario(scenario, tmp_path))
