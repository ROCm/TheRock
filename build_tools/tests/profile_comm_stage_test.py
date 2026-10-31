# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import profile_comm_stage as trial
import comm_stage_evidence as evidence
import compare_comm_stage as comparison


class CommStageTests(unittest.TestCase):
    def test_cache_identity_preserves_trust_domain_and_rejects_override(self):
        from setup_ccache import CACHE_SRV_DEV

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / ".ccache/ccache.conf"
            config.parent.mkdir()
            config.write_text("remote_storage = " + CACHE_SRV_DEV + "\n")
            source = root / "build_tools/posix_ccache_compiler_check.py"
            source.parent.mkdir()
            source.write_text("# exact compiler-check source\n")
            (config.parent / "compiler_check.py").write_bytes(source.read_bytes())
            values = dict(
                read_only="true", namespace="therock-v4", remote_storage=CACHE_SRV_DEV
            )
            with patch.object(trial, "ROOT", root), patch.object(
                trial,
                "capture",
                side_effect=lambda command, env: values.get(command[-1], "same"),
            ):
                trial.retain_cache_identity(
                    root / "out",
                    {"CCACHE_CONFIGPATH": str(config)},
                    time.monotonic() + 10,
                )
                values["remote_storage"] = (
                    "https://secret.invalid|password=never-export"
                )
                with self.assertRaisesRegex(RuntimeError, "identity drift"):
                    trial.retain_cache_identity(
                        root / "out",
                        {"CCACHE_CONFIGPATH": str(config)},
                        time.monotonic() + 10,
                    )
                self.assertNotIn(
                    "never-export", (root / "out/cache-identity.json").read_text()
                )

    @unittest.skipUnless(
        sys.platform == "linux", "POSIX command wait-chain/termination"
    )
    def test_command_cpu_scope_and_deadline_evidence_survive_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "successful.log"
            evidence.run_logged(
                [sys.executable, "-c", "sum(range(100000));print('complete')"],
                root,
                log,
                time.monotonic() + 10,
            )
            row = json.loads(log.with_suffix(".timing.json").read_text())
            self.assertEqual(row["exit_code"], 0)
            self.assertIn("complete", log.read_text())
            self.assertGreaterEqual(sum(row["waited_child_cpu"].values()), 0)
            self.assertGreaterEqual(row["controller_cpu_s"], 0)
            failed = root / "timeout.log"
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                evidence.run_logged(
                    [sys.executable, "-c", "import time;time.sleep(20)"],
                    root,
                    failed,
                    time.monotonic() + 0.2,
                )
            row = json.loads(failed.with_suffix(".timing.json").read_text())
            self.assertNotEqual(row["exit_code"], 0)
            self.assertLess(row["wall_s"], 8)

    def test_only_the_two_final_link_arguments_may_change(self):
        args = [
            "/sdk/clang++",
            "--hip-link",
            "-fgpu-rdc",
            *["--offload-arch=" + a for a in evidence.ARCHITECTURES],
            "input.o",
            "-o",
            evidence.TARGETS[0],
        ]
        evidence.validate_candidate(args, [*args, "--offload-jobs=2"])
        for bad in [
            [*args, "--offload-jobs=3"],
            [*args, "--offload-jobs=2", "-O0"],
            [*args, "--offload-jobs=2", "--offload-jobs=2"],
        ]:
            with self.assertRaises(RuntimeError):
                evidence.validate_candidate(args, bad)
        command = " ".join(args)
        self.assertEqual(evidence.driver_arguments(command, evidence.TARGETS[0]), args)
        for bad in [
            command + " -c",
            command + " @response.rsp",
            command.replace("gfx1250-strict", "gfx1250"),
        ]:
            with self.assertRaises(RuntimeError):
                evidence.driver_arguments(bad, evidence.TARGETS[0])
        self.assertEqual(len(evidence.RCCL_ARCHITECTURES), 25)

    def test_boundary_readonly_cpu_and_source_controls(self):
        env = dict(RELEASE_TYPE="ci", CCACHE_READONLY="1", TEATIME_PROFILE_CPU="1")

        def git(args, env, cwd=trial.ROOT):
            if args[0] == "rev-parse":
                return trial.SYSTEMS if args[1].endswith("rocm-systems") else trial.LLVM
            return ""

        with patch.object(evidence, "require_rock_ci"), patch.object(
            trial.sys, "platform", "linux"
        ), patch.object(trial, "git", side_effect=git):
            trial.require_controls(env, 1)
            for overrides in [
                {"CCACHE_READONLY": "0"},
                {"RELEASE_TYPE": "nightly"},
                {"TEATIME_PROFILE_CPU": "0"},
                {"HIP_CLANG_NUM_PARALLEL_JOBS": "2"},
            ]:
                with self.assertRaises(RuntimeError):
                    trial.require_controls({**env, **overrides}, 2)
        for repo in ["ROCm/rockrel", "other/TheRock"]:
            with patch.dict(
                os.environ, dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=repo)
            ):
                with self.assertRaisesRegex(RuntimeError, "ROCm/TheRock CI"):
                    evidence.require_rock_ci()

    def test_exact_cpu_reconciliation_does_not_scale_samples_or_hide_missing_phases(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            output = build / "out"
            (build / "logs").mkdir()
            (output / "profile").mkdir(parents=True)
            evidence.save(
                output / "profile/summary.json",
                dict(
                    posix_waited_child_cpu=dict(user_s=120, system_s=7),
                    sampled_process_cpu_lower_bound_s=30,
                ),
            )
            record = dict(
                child_total_s=6,
                child_user_s=5,
                child_system_s=1,
                exit_code=0,
                phase_wall_s=2,
            )
            for name in evidence.PROJECTS:
                for phase in ["configure", "build", "install"]:
                    (build / "logs" / f"{name}_{phase}.log").write_text(
                        "native output\nCPU_PROFILE\t" + json.dumps(record) + "\n"
                    )
            result = evidence.collect_phase_cpu(build, output)
            self.assertEqual(result["record_count"], 21)
            self.assertEqual(result["phase_waited_child_cpu_s"], 126)
            self.assertEqual(result["stage_waited_child_cpu_s"], 127)
            self.assertEqual(result["residual_unallocated_cpu_s"], 1)
            (build / "logs/rocshmem_build.log").unlink()
            with self.assertRaises(RuntimeError):
                evidence.collect_phase_cpu(build, output)
            self.assertEqual(
                json.loads((output / "phase-cpu.json").read_text())["record_count"], 20
            )

    def test_independent_collectors_continue_after_a_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                evidence, "collect_phase_cpu", side_effect=ValueError("metadata")
            ), patch.object(evidence, "collect_graphs"), patch.object(
                trial.auxiliary, "retain"
            ), patch.object(
                evidence, "retain_link_inputs"
            ), patch.object(
                evidence, "retain_outputs", return_value={}
            ), patch.object(
                evidence, "retain_rccl"
            ) as rccl, patch.object(
                trial, "collect_artifacts"
            ), patch.object(
                trial, "packaging_tests"
            ) as ctest:
                result = trial.postchecks(root, root, 1, {}, time.monotonic() + 10)
            self.assertEqual(result[0]["status"], "failed")
            self.assertEqual(len(result), 8)
            self.assertTrue(all(r["status"] == "passed" for r in result[1:]))
            rccl.assert_called_once()
            ctest.assert_called_once()
            self.assertTrue((root / "postchecks.json").is_file())

    def fixture_pair(self, base):
        roots = [base / "A", base / "B"]
        common = dict(
            complete=True,
            source_sha="same",
            product_head=trial.PRODUCT,
            reference_head=trial.BASE,
            sdk_run=trial.SDK_RUN,
            sdk_repository="ROCm/TheRock",
            version=trial.VERSION,
            runner_label="aws-linux-scale-rocm-prod",
            container_digest="d666",
            cache_controls={},
        )
        for index, root in enumerate(roots, 1):
            root.mkdir()
            evidence.save(root / "manifest.json", dict(common, mode=index))
            for name in [
                "source-identity.json",
                "cache-identity.json",
                "link-inputs.json",
                "rocshmem-native.json",
                "rccl-native.json",
            ]:
                evidence.save(root / name, {"same": "data"})
            for name in [
                "python-packages.json",
                "tool-versions.json",
                "artifact-files.jsonl",
                "stage-cmake-args.txt",
            ]:
                (root / name).write_text("unchanged\n")
            evidence.save(
                root / "configure-command.json",
                ["cmake", f"-DTHEROCK_ROCSHMEM_NATIVE_LINK_JOBS={index}"],
            )
            edges = {
                name: ["clang++", "input.o", "-o", name]
                + (["--offload-jobs=2"] if index == 2 else [])
                for name in evidence.TARGETS
            }
            evidence.save(root / "final-link-edges.json", edges)
            for project in evidence.PROJECTS:
                directory = root / "graphs" / project
                directory.mkdir(parents=True)
                (directory / "generated-and-object-manifest.jsonl").write_text(
                    '{"path":"obj.o","sha256":"original"}\n'
                )
                (directory / "compile_commands.json").write_text("[]")
                evidence.save(
                    directory / "all-edges.json",
                    [
                        dict(
                            output="custom.o",
                            command="hipcc -c source.cpp",
                            directory="same",
                        )
                    ],
                )
            evidence.save(
                root / "profile/metadata.json",
                dict(
                    visible_logical_cpus=96,
                    affinity=[0, 1],
                    effective_capacity_visible_upper_bound_cpus=2,
                ),
            )
        return roots

    def test_parity_rejects_changed_objects_custom_edges_flags_or_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = self.fixture_pair(Path(temporary))
            self.assertTrue(comparison.compare(a, b)["input_and_artifact_parity"])
            manifest = b / "graphs/rccl/generated-and-object-manifest.jsonl"
            original = manifest.read_text()
            manifest.write_text(original.replace("original", "modified"))
            self.assertFalse(comparison.compare(a, b)["input_and_artifact_parity"])
            manifest.write_text(original)
            edges = b / "graphs/rccl/all-edges.json"
            original = edges.read_text()
            rows = json.loads(original)
            rows[0]["command"] += " -O0"
            evidence.save(edges, rows)
            result = comparison.compare(a, b)
            self.assertTrue(any("custom-compile" in e for e in result["errors"]))
            edges.write_text(original)
            final = b / "final-link-edges.json"
            rows = json.loads(final.read_text())
            rows[evidence.TARGETS[0]].append("-O0")
            evidence.save(final, rows)
            self.assertFalse(comparison.compare(a, b)["input_and_artifact_parity"])
            self.assertFalse(comparison.compare(a, b)["performance_validated"])

    def test_workflow_no_publication_and_fresh_full_build(self):
        source = (trial.ROOT / ".github/workflows/multi_arch_ci.yml").read_text()
        job = source.split("\n  profile_comm_stage:\n", 1)[1].split(
            "\n  linux_build_and_test:", 1
        )[0]
        for required in [
            "github.repository == 'ROCm/TheRock'",
            "runs-on: aws-linux-scale-rocm-prod",
            "CCACHE_READONLY: '1'",
            "TEATIME_PROFILE_CPU: '1'",
            "include-hidden-files: true",
            '--log-dir "$GITHUB_WORKSPACE/.container-cache/ccache-logs" --init',
        ]:
            self.assertIn(required, job)
        for forbidden in [
            "id-token:",
            "secrets:",
            "upload_package",
            "configure_aws",
            "git config --global",
            "gh auth setup-git",
        ]:
            self.assertNotIn(forbidden, job)
        implementation = (trial.ROOT / "build_tools/profile_comm_stage.py").read_text()
        self.assertNotIn('"rocshmem+configure"', implementation)
        self.assertNotIn('"HIP_CLANG_NUM_PARALLEL_JOBS=2"', implementation)
        self.assertIn('"stage-comm-libs"', implementation)
        self.assertIn('"therock-artifacts"', implementation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
