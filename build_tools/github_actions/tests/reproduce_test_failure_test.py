# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import reproduce_test_failure


class ReproduceTestFailureTest(unittest.TestCase):
    def _make_args(
        self,
        *,
        amdgpu_targets: str = "gfx942",
        shard_index: str = "2",
        total_shards: str = "4",
        test_type: str = "quick",
        fetch_artifact_args: str = "--blas --tests",
        additional_requirements_files: str = (
            "share/hipblaslt/tensilelite/requirements-test.txt"
        ),
    ) -> argparse.Namespace:
        return argparse.Namespace(
            run_id="1234",
            repository="ROCm/TheRock",
            amdgpu_family="gfx94X-dcgpu",
            amdgpu_targets=amdgpu_targets,
            test_script="python test.py",
            shard_index=shard_index,
            total_shards=total_shards,
            test_type=test_type,
            container_image="test-image",
            fetch_artifact_args=fetch_artifact_args,
            additional_requirements_files=additional_requirements_files,
            setup_only=False,
            print_cmd=False,
        )

    def test_build_reproduction_command_includes_non_default_arguments(self):
        command = reproduce_test_failure.build_reproduction_command(self._make_args())

        self.assertEqual(
            command,
            "python build_tools/github_actions/reproduce_test_failure.py "
            "--run-id 1234 --repository ROCm/TheRock "
            '--amdgpu-family gfx94X-dcgpu --test-script "python test.py" '
            "--amdgpu-targets gfx942 --shard-index 2 --total-shards 4 "
            '--test-type quick --fetch-artifact-args="--blas --tests" '
            '--additional-requirements-files="share/hipblaslt/'
            'tensilelite/requirements-test.txt"',
        )

    def test_build_reproduction_command_omits_default_arguments(self):
        command = reproduce_test_failure.build_reproduction_command(
            self._make_args(
                amdgpu_targets="",
                shard_index="1",
                total_shards="1",
                test_type="full",
                fetch_artifact_args="",
                additional_requirements_files="",
            )
        )

        self.assertEqual(
            command,
            "python build_tools/github_actions/reproduce_test_failure.py "
            "--run-id 1234 --repository ROCm/TheRock "
            '--amdgpu-family gfx94X-dcgpu --test-script "python test.py"',
        )

    def test_run_linux_runs_ordered_setup_and_test_steps(self):
        with (
            mock.patch.object(
                reproduce_test_failure, "check_docker", return_value=True
            ),
            mock.patch.object(
                reproduce_test_failure.subprocess,
                "run",
                return_value=mock.Mock(returncode=0),
            ) as run,
        ):
            result = reproduce_test_failure.run_linux(self._make_args())

        self.assertEqual(result, 0)
        run.assert_called_once()
        docker_command = run.call_args.args[0]
        self.assertEqual(docker_command[:2], ["docker", "run"])
        self.assertEqual(
            docker_command[-4:-1],
            [
                "test-image",
                "/bin/bash",
                "-c",
            ],
        )

        script = docker_command[-1]
        script_lines = script.splitlines()
        progress_lines = [line for line in script_lines if line.startswith("echo '[")]
        step_descriptions = [
            line.partition("] ")[2].removesuffix("'") for line in progress_lines
        ]
        self.assertEqual(
            step_descriptions,
            [
                "Installing uv",
                "Cloning TheRock",
                "Creating virtual environment",
                "Installing dependencies",
                "Downloading artifacts",
                "Setting environment variables",
                "Installing component test dependencies",
                "Running test",
            ],
        )

        artifact_command = next(
            line for line in script_lines if "install_rocm_from_artifacts.py" in line
        )
        self.assertIn("GITHUB_REPOSITORY=ROCm/TheRock", artifact_command)
        self.assertIn("--run-id 1234", artifact_command)
        self.assertIn("--amdgpu-family gfx94X-dcgpu", artifact_command)
        self.assertIn("--amdgpu-targets gfx942", artifact_command)
        self.assertIn("--blas --tests", artifact_command)

        requirements_command = next(
            line
            for line in script_lines
            if "install_additional_requirements.py" in line
        )
        self.assertIn(
            "share/hipblaslt/tensilelite/requirements-test.txt",
            requirements_command,
        )

        environment_command = next(
            line for line in script_lines if "export THEROCK_BIN_DIR" in line
        )
        self.assertIn("export SHARD_INDEX=2", environment_command)
        self.assertIn("export TOTAL_SHARDS=4", environment_command)
        self.assertIn("export TEST_TYPE=quick", environment_command)
        self.assertTrue(any(line.startswith("python test.py") for line in script_lines))


if __name__ == "__main__":
    unittest.main()
