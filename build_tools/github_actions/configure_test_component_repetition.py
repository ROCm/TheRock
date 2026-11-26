# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure component test repetition matrices and a shared artifact baseline."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from amdgpu_family_matrix import (
    get_all_families_for_trigger_types,
    load_external_runner_config,
)
from fetch_package_targets import determine_package_targets
from github_actions_api import gha_set_output

# Each family/platform has a separate caller job so its repetition matrix can
# execute serially without serializing the other triage targets.
REPETITION_FAMILIES = {
    "gfx120x": "gfx120X-all",
    "gfx110x": "gfx110X-all",
    "gfx1151": "gfx1151",
}


def _csv_values(value: str) -> list[str]:
    return list(dict.fromkeys(v.strip() for v in value.split(",") if v.strip()))


def _runner_labels(value: object) -> set[str]:
    """Collect active runner labels from an external pool definition."""
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, list):
        return set().union(*(_runner_labels(v) for v in value))
    if isinstance(value, dict) and value.get("label") and value.get("count", 1) > 0:
        return {value["label"]}
    if isinstance(value, dict) and "label" not in value:
        return set().union(*(_runner_labels(v) for v in value.values()))
    return set()


def fetch_repetition_components(
    *, family: str, platform: str, environment: dict[str, str]
) -> list[dict]:
    """Run the existing configuration CLI with isolated workflow outputs."""
    script = Path(__file__).with_name("fetch_test_configurations.py")
    with tempfile.TemporaryDirectory(prefix="component-repetition-") as directory:
        output = Path(directory) / "outputs"
        env = dict(environment)
        # The reusable workflow consumes per-component routing; do not apply
        # test_artifacts.yml's manual GPU runner override to these components.
        env.pop("TEST_RUNS_ON", None)
        env["TEST_RUNS_ON_CPU"] = ""
        env.update(
            AMDGPU_FAMILIES=family,
            TEST_TYPE=environment.get("TEST_TYPE", "comprehensive"),
            PROJECTS_TO_TEST="*",
            GITHUB_OUTPUT=str(output),
        )
        subprocess.run(
            [sys.executable, str(script), "--platform", platform],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            check=True,
        )
        outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
        components = json.loads(outputs["components"])
        # Sanity is normally a separate prerequisite. Include it only when
        # explicitly selected, rather than adding it to every component sample.
        from fetch_test_configurations import _parse_test_labels

        if "test:sanity" in _parse_test_labels(env.get("TEST_LABELS", "")):
            sanity = json.loads(outputs["sanity_component"])
            if sanity:
                components.insert(0, sanity)
        return components


def create_repetition_plan(environment: dict[str, str]) -> tuple[dict, dict]:
    """Resolve a bounded pilot using the existing package and test helpers.

    Required environment: AMDGPU_FAMILIES, TEST_LABELS, CI_CONFIG_PATH.
    Optional: PLATFORMS (empty means both), REPETITIONS (20), AMDGPU_TARGETS,
    BUILD_VARIANT (release), TEST_TYPE (comprehensive).
    CI_CONFIG_PATH must be a pinned triage checkout.
    """
    from fetch_test_configurations import (
        _parse_test_labels,
        test_matrix,
        TEST_LABEL_GROUPS,
    )

    repetitions = int(environment.get("REPETITIONS", "20"))
    test_type = environment.get("TEST_TYPE", "comprehensive")
    if test_type not in ("quick", "standard", "comprehensive", "full"):
        raise ValueError("TEST_TYPE must be quick, standard, comprehensive, or full")
    if not 1 <= repetitions <= 100:
        raise ValueError("REPETITIONS must be between 1 and 100")
    if environment.get("BUILD_VARIANT", "release") != "release":
        raise ValueError(
            "The repetition pilot currently supports release artifacts only"
        )
    families = _csv_values(environment.get("AMDGPU_FAMILIES", ""))
    if not families:
        raise ValueError("Select at least one AMDGPU family")
    keys = []
    for family in families:
        key = family.lower().split("-")[0]
        if key not in REPETITION_FAMILIES or family.lower() not in (
            key,
            REPETITION_FAMILIES[key].lower(),
        ):
            raise ValueError(f"Unsupported pilot family: {family}")
        if key not in keys:
            keys.append(key)
    if environment.get("AMDGPU_TARGETS") and len(keys) != 1:
        raise ValueError("AMDGPU_TARGETS requires a single family")
    platforms = _csv_values(environment.get("PLATFORMS", "")) or ["linux", "windows"]
    if any(p not in ("linux", "windows") for p in platforms):
        raise ValueError("PLATFORMS must contain linux and/or windows")
    labels = _parse_test_labels(environment.get("TEST_LABELS", ""))
    known_labels = set(test_matrix) | set(TEST_LABEL_GROUPS)
    if not labels or any(
        not label.startswith("test:") or label[5:] not in known_labels
        for label in labels
    ):
        raise ValueError("TEST_LABELS must select known test: component labels")

    config_path = Path(environment.get("CI_CONFIG_PATH", ""))
    if (
        not environment.get("CI_CONFIG_PATH")
        or not (config_path / "ci_config_api.py").is_file()
    ):
        raise ValueError("CI_CONFIG_PATH must contain the custom CI config API")
    external = load_external_runner_config()
    if external is None:
        raise ValueError(
            "Custom CI config could not be loaded; refusing local runner fallback"
        )
    sha = subprocess.check_output(
        ["git", "-C", str(config_path), "rev-parse", "HEAD"], text=True
    ).strip()
    matrices = {
        f"{platform}_{key}": {"include": []}
        for platform in ("linux", "windows")
        for key in REPETITION_FAMILIES
    }
    expected = []
    targets = []
    all_families = get_all_families_for_trigger_types(
        ["presubmit", "postsubmit", "nightly"]
    )
    for platform in platforms:
        package_targets = determine_package_targets(
            {"AMDGPU_FAMILIES": ",".join(keys), "THEROCK_PACKAGE_PLATFORM": platform}
        )
        resolved = {target["amdgpu_family"]: target for target in package_targets}
        for key in keys:
            family = REPETITION_FAMILIES[key]
            if family not in resolved:
                raise ValueError(f"No package target for {family}/{platform}")
            configured = (
                external.get("runner_labels", {}).get(key, {}).get(platform, {})
            )
            allowed = set().union(
                *(
                    _runner_labels(value)
                    for name, value in configured.items()
                    if name.startswith("test-runs-on")
                )
            )
            cpu_label = external.get("cpu_test_runners", {}).get(platform, "")
            if not allowed:
                raise ValueError(
                    f"No explicit custom GPU runner mapping for {family}/{platform}"
                )
            gpu_targets = _csv_values(environment.get("AMDGPU_TARGETS", ""))
            if gpu_targets and not set(gpu_targets).issubset(
                all_families[key][platform].get("fetch-gfx-targets", [])
            ):
                raise ValueError(
                    f"AMDGPU_TARGETS are not configured for {family}/{platform}"
                )
            components = [
                dict(component)
                for component in fetch_repetition_components(
                    family=family, platform=platform, environment=environment
                )
            ]
            if not components:
                raise ValueError(f"No selected components for {family}/{platform}")
            if repetitions * len(components) > 256:
                raise ValueError(
                    f"{family}/{platform}: repetitions × components exceeds 256"
                )
            target_key = f"{platform}_{key}"
            matrices[target_key] = {
                "iteration": list(range(1, repetitions + 1)),
                "component": components,
            }
            targets.append(
                {"family": family, "platform": platform, "components": components}
            )
            for component in components:
                # Artifacts belong to rockrel, while the test harness checkout
                # belongs to TheRock. The installer otherwise queries TheRock
                # for the foreign run ID and fails before tests can execute.
                component["fetch_artifact_args"] += " --run-github-repo ROCm/rockrel"
                name = component["job_name"]
                if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                    raise ValueError(f"Unsupported component job name: {name}")
                runner = component.get("multi_gpu_runner") or component.get(
                    "test_runner"
                )
                permitted = (
                    {cpu_label} if component.get("linux_cpu_runner") else allowed
                )
                if not runner or runner not in permitted:
                    raise ValueError(
                        f"{family}/{platform}/{name}: runner {runner!r} is not explicitly mapped in custom CI config"
                    )
                if component.get("test_type") != test_type:
                    raise ValueError(
                        f"{family}/{platform}/{name}: effective tier must be {test_type}"
                    )
                shards = component["shard_arr"]
                if not shards or shards != list(
                    range(1, component["total_shards"] + 1)
                ):
                    raise ValueError(f"Invalid shard configuration for {name}")
                for iteration in range(1, repetitions + 1):
                    sample_id = f"repeat-{platform}-{key}-{iteration}-{name}"
                    for shard in shards:
                        expected.append(
                            {
                                "sample_id": sample_id,
                                "family": family,
                                "platform": platform,
                                "component": name,
                                "iteration": iteration,
                                "shard": shard,
                                "total_shards": component["total_shards"],
                                "runner_label": runner,
                                "expect_failure": component.get(
                                    "expect_failure", False
                                ),
                            }
                        )
    return matrices, {
        "schema_version": 1,
        "repository": environment.get("GITHUB_REPOSITORY", ""),
        "therock_sha": environment.get("GITHUB_SHA", ""),
        "ci_config_sha": sha,
        "test_type": test_type,
        "build_variant": "release",
        "test_labels": labels,
        "amdgpu_targets": environment.get("AMDGPU_TARGETS", ""),
        "repetitions": repetitions,
        "targets": targets,
        "expected": expected,
    }


def write_repetition_outputs(environment: dict[str, str], manifest_path: Path) -> None:
    from baseline_runs import query_successful_workflow_runs, is_successful_workflow_run

    matrices, manifest = create_repetition_plan(environment)
    runs = query_successful_workflow_runs(
        github_repository="ROCm/rockrel",
        workflow_name="multi_arch_release.yml",
        branch="main",
        max_runs=100,
    )
    baseline = next(
        (
            run
            for run in runs
            if run.get("event") == "schedule" and is_successful_workflow_run(run)
        ),
        None,
    )
    if baseline is None:
        raise ValueError(
            "No successful scheduled rockrel nightly found in the latest 100 successful runs"
        )
    manifest["baseline"] = {
        "repository": "ROCm/rockrel",
        "run_id": str(baseline["id"]),
        "head_sha": baseline["head_sha"],
        "url": baseline["html_url"],
        "created_at": baseline["created_at"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    outputs = {
        key: json.dumps(value, separators=(",", ":")) for key, value in matrices.items()
    }
    # Actions limits all job outputs to approximately 1 MB, measured as UTF-16.
    if sum(len(value.encode("utf-16-le")) for value in outputs.values()) > 900_000:
        raise ValueError(
            "Repetition matrices exceed the workflow job output size budget"
        )
    gha_set_output(
        {
            **outputs,
            "artifact_run_id": str(baseline["id"]),
            "enabled_targets": json.dumps(
                [key for key, matrix in matrices.items() if "component" in matrix]
            ),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("component-repetition/manifest.json"),
        help="Path to write the resolved repetition manifest",
    )
    options = parser.parse_args()
    write_repetition_outputs(dict(os.environ), options.manifest)
