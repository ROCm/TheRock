# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "profile_offload_trial.py"
spec = importlib.util.spec_from_file_location("profile_offload_trial", SCRIPT)
trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


def ninja_text() -> str:
    return (
        "rule compiler\n  command = clang++ $FLAGS -c $in -o $out\n\n"
        f"build {trial.OBJECT}: compiler source.cpp\n"
        "  FLAGS = -O3 "
        + " ".join(f"--offload-arch={arch}" for arch in sorted(trial.ARCHES))
        + "\n"
        + f"  DEP_FILE = {trial.OBJECT}.d\n\n"
        + f"build {trial.OBJECT}.neighbor: compiler neighbor.cpp\n"
        + "  FLAGS = -O3\n\n"
        + "build untouched: phony\n"
    )


def compiler_command(jobs: int = 2) -> str:
    return (
        "ccache /sdk/bin/clang++ -O3 "
        + " ".join(f"--offload-arch={arch}" for arch in sorted(trial.ARCHES))
        + f" --offload-jobs={jobs} -c /source/benchmark_device_segmented_radix_sort_pairs.cpp"
        + f" -o {trial.OBJECT}"
    )


class TrialGuardsTest(unittest.TestCase):
    def test_one_exact_object_edge_changes_and_neighbor_is_untouched(self):
        original = ninja_text()
        for jobs in (1, 2):
            with self.subTest(jobs=jobs):
                patched = trial.patch_flags(original, jobs)
                self.assertEqual(
                    patched.replace(f" --offload-jobs={jobs}", ""), original
                )
                self.assertEqual(patched.count("--offload-jobs="), 1)
                self.assertEqual(patched.splitlines()[-4:], original.splitlines()[-4:])
                self.assertEqual(
                    {a for a in trial.ARCHES if f"--offload-arch={a}" in patched},
                    trial.ARCHES,
                )

    def test_ambiguous_or_previously_modified_edges_fail_before_execution(self):
        original = ninja_text()
        for text in (
            original.replace(trial.OBJECT + ":", trial.OBJECT + ".wrong:"),
            original + original,
            original.replace("FLAGS = -O3 ", "FLAGS = -O3 --offload-jobs=4 ", 1),
            original.replace("FLAGS = -O3 ", "FLAGS = -O3 -parallel-jobs=2 ", 1),
            original.replace("  FLAGS = -O3 ", "  NOT_FLAGS = -O3 ", 1),
            original.replace("  DEP_FILE", "    FLAGS = -O0\n  DEP_FILE", 1),
        ):
            with self.subTest(text=text[:50]):
                with self.assertRaises(ValueError):
                    trial.patch_flags(text, 2)
        with self.assertRaises(ValueError):
            trial.patch_flags(original, 4)

    def test_commands_keep_seven_targets_and_reject_silent_serial_or_noop_modes(self):
        compiler, args = trial.compiler_arguments(compiler_command(), 2)
        self.assertEqual(compiler.as_posix(), "/sdk/bin/clang++")
        self.assertIn("--offload-jobs=2", args)
        for bad in (
            compiler_command().replace("--offload-arch=gfx1036", ""),
            compiler_command().replace("--offload-jobs=2", "--offload-jobs=1"),
            compiler_command() + " --offload-jobs=2",
            compiler_command() + " -parallel-jobs=2",
            compiler_command() + " -v",
            compiler_command() + " -fproc-stat-report=stats.csv",
            compiler_command() + " --driver-mode=cl",
            compiler_command() + " -###",
            compiler_command() + " -fdriver-only",
            compiler_command() + " && echo hidden-command",
            compiler_command().replace(
                "benchmark_device_segmented_radix_sort_pairs.cpp", "other.cpp"
            ),
        ):
            with self.subTest(command=bad[-60:]):
                with self.assertRaises(ValueError):
                    trial.compiler_arguments(bad, 2)

    def test_cache_and_profiler_environment_cannot_override_the_trial(self):
        original = {key: "sensitive-value-not-to-log" for key in trial.DISABLING_ENV}
        original.update(
            CCACHE_DISABLE="0",
            GH_TOKEN="not-a-real-token",
            AWS_SECRET_ACCESS_KEY="not-a-real-key",
            PATH="/sdk/bin",
            LD_LIBRARY_PATH="/sdk/lib",
        )
        cleaned = trial.trial_env(original)
        self.assertEqual(cleaned["CCACHE_DISABLE"], "1")
        self.assertFalse(trial.DISABLING_ENV.intersection(cleaned))
        self.assertNotIn("GH_TOKEN", cleaned)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", cleaned)
        self.assertEqual(cleaned["LD_LIBRARY_PATH"], "/sdk/lib")
        self.assertEqual(original["CCACHE_DISABLE"], "0")

    def test_only_explicit_balanced_orders_and_therock_ci_are_accepted(self):
        for order in ([1, 2], [2, 1]):
            self.assertEqual(trial.parse_order(json.dumps(order)), order)
        for order in ([1, 1], [1, 2, 4], [True, 2], ["1", "2"], None):
            with self.assertRaises((TypeError, ValueError)):
                trial.parse_order(json.dumps(order))
        trial.require_ci(
            {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "ROCm/TheRock"}
        )
        for env in (
            {},
            {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "ROCm/rockrel"},
        ):
            with self.assertRaises(RuntimeError):
                trial.require_ci(env)

    def test_force_rebuild_unlinks_only_generated_object_and_depfile(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            paths = [
                trial.generated_path(build, name)
                for name in (
                    trial.OBJECT,
                    trial.OBJECT + ".d",
                    trial.OBJECT + ".neighbor",
                )
            ]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("generated")
            removed = trial.force_object_rebuild(build)
            self.assertEqual(set(removed), {trial.OBJECT, trial.OBJECT + ".d"})
            self.assertTrue(paths[2].exists())
            self.assertFalse(paths[0].exists())
            self.assertFalse(paths[1].exists())
            self.assertEqual(trial.force_object_rebuild(build), [])
            for path in ("../outside", str(build.resolve() / "absolute")):
                with self.assertRaises(ValueError):
                    trial.generated_path(build, path)

    def test_symlinked_outputs_cannot_delete_source_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / "build"
            build.mkdir()
            source = root / "source.cpp"
            source.write_text("source must survive")
            output = build / "object.o"
            try:
                output.symlink_to(source)
            except OSError:
                self.skipTest("Host does not permit symlink creation")
            with self.assertRaises(ValueError):
                trial.force_object_rebuild(build, "object.o")
            self.assertEqual(source.read_text(), "source must survive")


@unittest.skipUnless(
    shutil.which("ninja"), "Ninja is required for incremental-build verification"
)
class NativeNinjaTest(unittest.TestCase):
    def test_balanced_orders_force_exactly_one_compile_and_reuse_it_for_linking(self):
        # Real Ninja command hashes and dependency decisions, with a tiny Python
        # producer instead of a ROCm build. Removing FLAGS scoping or the forced
        # rebuild, or recompiling the object during link, fails these assertions.
        for order in ([1, 2], [2, 1]):
            with self.subTest(order=order), tempfile.TemporaryDirectory() as temporary:
                build = Path(temporary)
                writer = build / "writer.py"
                writer.write_text(
                    "from pathlib import Path\nimport sys\n"
                    "mode,output,counter,*flags=sys.argv[1:]\n"
                    "p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)\n"
                    "c=Path(counter);c.write_text(str(int(c.read_text())+1) if c.exists() else '1')\n"
                    "if mode=='compile':\n p.write_text(' '.join(flags));Path(output+'.d').write_text('generated depfile')\n"
                    "else:\n p.write_bytes(Path(flags[0]).read_bytes())\n",
                    encoding="utf-8",
                )
                python = sys.executable.replace("\\", "/")
                helper = str(writer).replace("\\", "/")
                original = (
                    "rule compiler\n"
                    f'  command = "{python}" "{helper}" compile $out compile-count $FLAGS\n'
                    "rule linker\n"
                    f'  command = "{python}" "{helper}" link $out link-count $in\n\n'
                    f"build {trial.OBJECT}: compiler\n"
                    "  FLAGS = -O3 "
                    + " ".join(f"--offload-arch={a}" for a in sorted(trial.ARCHES))
                    + "\n\n"
                    f"build {trial.BENCHMARK}: linker {trial.OBJECT}\n"
                )
                manifest = build / "build.ninja"
                hashes = []
                for i, jobs in enumerate(order, 1):
                    manifest.write_text(
                        trial.patch_flags(original, jobs), encoding="utf-8"
                    )
                    trial.force_object_rebuild(build)
                    subprocess.run(
                        ["ninja", "-C", str(build), trial.OBJECT],
                        check=True,
                        capture_output=True,
                    )
                    command = trial.single_command(
                        "ninja", build, trial.OBJECT, dict(os.environ)
                    )
                    self.assertIn(f"--offload-jobs={jobs}", command)
                    before = trial.last_object_record(build)
                    output = trial.generated_path(build, trial.OBJECT)
                    state = (output.stat().st_mtime_ns, trial.digest(output))
                    subprocess.run(
                        ["ninja", "-C", str(build), trial.BENCHMARK],
                        check=True,
                        capture_output=True,
                    )
                    self.assertEqual(
                        state, (output.stat().st_mtime_ns, trial.digest(output))
                    )
                    self.assertEqual(before, trial.last_object_record(build))
                    self.assertEqual((build / "compile-count").read_text(), str(i))
                    self.assertEqual((build / "link-count").read_text(), str(i))
                    hashes.append(before.split("\t")[-1])
                self.assertNotEqual(*hashes)


class WorkflowContractTest(unittest.TestCase):
    def test_opt_in_mode_is_paired_pinned_and_has_no_release_side_effects(self):
        workflow = yaml.load(
            (SCRIPT.parents[1] / ".github/workflows/multi_arch_ci.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        for trigger in ("workflow_call", "workflow_dispatch"):
            self.assertEqual(
                workflow["on"][trigger]["inputs"]["profile_offload_trial"]["default"],
                "false",
            )
        jobs = workflow["jobs"]
        for name in (
            "setup",
            "linux_build_and_test",
            "windows_build_and_test",
            "manifest_diff",
            "ci_summary",
        ):
            self.assertIn("!inputs.profile_offload_trial", jobs[name]["if"])
        experiment = jobs["profile_offload_trial"]
        self.assertIn("github.repository == 'ROCm/TheRock'", experiment["if"])
        self.assertEqual(experiment["runs-on"], "aws-linux-scale-rocm-prod")
        self.assertEqual(experiment["timeout-minutes"], "360")
        self.assertEqual(
            experiment["strategy"]["matrix"]["order"], [["1", "2"], ["2", "1"]]
        )
        self.assertIn(
            "@sha256:d666e4613c1ca919bc7d1a31a4c368740b66c34b5c19ef95ba15cd31db7783fb",
            experiment["container"]["image"],
        )
        text = json.dumps(experiment)
        self.assertNotIn("id-token", text)
        self.assertNotIn("configure_aws", text)
        self.assertNotIn("artifact_manager.py push", text)
        upload = experiment["steps"][-1]
        self.assertEqual(upload["if"], "always()")
        self.assertTrue(upload["uses"].startswith("actions/upload-artifact@"))


if __name__ == "__main__":
    unittest.main()
