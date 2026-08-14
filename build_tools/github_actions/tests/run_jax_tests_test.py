"""
Unit tests for external-builds/jax/run_jax_tests.py
"""

import json
import os
import platform
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent
JAX_DIR = THIS_DIR.parents[2] / "external-builds" / "jax"

sys.path.insert(0, os.fspath(JAX_DIR))

import run_jax_tests as runner

GFX94_FAMILY = "gfx94X-dcgpu"


class JaxVersionFromRefTest(unittest.TestCase):
    def test_strips_the_ref_prefix(self):
        self.assertEqual(runner.jax_version_from_ref("rocm-jaxlib-v0.11.0"), "0.11.0")

    def test_leaves_a_bare_version_alone(self):
        self.assertEqual(runner.jax_version_from_ref("0.11.0"), "0.11.0")

    def test_empty_ref(self):
        self.assertEqual(runner.jax_version_from_ref(""), "")


class PytestAddoptsTest(unittest.TestCase):
    def test_in_process_reruns(self):
        opts = shlex.split(runner.pytest_addopts("0.11.0", GFX94_FAMILY, 2))

        self.assertEqual(
            opts, ["--reruns", "2", "--reruns-delay", "5", "--only-rerun", "INTERNAL"]
        )

    def test_zero_reruns_leaves_them_out(self):
        self.assertEqual(runner.pytest_addopts("0.11.0", GFX94_FAMILY, 0), "")

    def test_skip_list_is_one_quoted_argument(self):
        # The expression contains spaces and parens, so it has to survive the
        # shlex split pytest performs on PYTEST_ADDOPTS as a single -k value.
        opts = shlex.split(runner.pytest_addopts("0.10.2", GFX94_FAMILY, 0))

        self.assertEqual(opts[0], "-k")
        self.assertEqual(len(opts), 2)
        self.assertIn("not conv", opts[1])

    def test_reruns_and_skip_list_together(self):
        opts = shlex.split(runner.pytest_addopts("0.10.2", GFX94_FAMILY, 2))

        self.assertEqual(opts[0], "--reruns")
        self.assertEqual(opts[-2], "-k")


SUITE_SCRIPT_TEMPLATE = """\
#!/bin/bash
source ci/envs/default.env
echo "pretend to install wheels"
export SHOULD_NOT_LEAK=installs-above-the-section

# ==============================================================================
# Set up the generic test environment variables
# ==============================================================================
export JAX_SKIP_SLOW_TESTS=true
export XLA_PYTHON_CLIENT_ALLOCATOR={allocator}
export JAX_ENABLE_X64="$JAXCI_ENABLE_X64"
export gpu_count=2
export JAX_ENABLE_ROCM_XDIST="$gpu_count"
if [[ -n "{conditional}" ]]; then
  export XLA_FLAGS="--xla_gpu_enable_cublaslt=false"
else
  export XLA_FLAGS="--xla_gpu_enable_command_buffer="
fi

# ==============================================================================
# Run tests
# ==============================================================================
export SHOULD_NOT_LEAK=below-the-section
exit 1
"""


def write_suite_script(jax_dir: Path, allocator="address", conditional="yes") -> None:
    """A checkout from before the environment moved out of the suite script."""
    (jax_dir / "ci" / "envs").mkdir(parents=True, exist_ok=True)
    (jax_dir / "ci" / "envs" / "default.env").write_text(
        "export JAXCI_ENABLE_X64=${JAXCI_ENABLE_X64:-0}\n"
    )
    (jax_dir / runner.RELATIVE_SUITE_SCRIPT).write_text(
        SUITE_SCRIPT_TEMPLATE.format(allocator=allocator, conditional=conditional)
    )


def write_split_suite_script(
    jax_dir: Path, allocator="address", conditional="yes"
) -> None:
    """A checkout as it is now, with the environment in its own file."""
    write_suite_script(jax_dir, allocator=allocator, conditional=conditional)
    section = runner.env_section((jax_dir / runner.RELATIVE_SUITE_SCRIPT).read_text())
    (jax_dir / runner.RELATIVE_SUITE_ENV_SCRIPT).parent.mkdir(
        parents=True, exist_ok=True
    )
    (jax_dir / runner.RELATIVE_SUITE_ENV_SCRIPT).write_text(section)
    (jax_dir / runner.RELATIVE_SUITE_SCRIPT).write_text(
        "#!/bin/bash\nsource ci/envs/default.env\n"
        f"source {runner.RELATIVE_SUITE_ENV_SCRIPT.as_posix()}\nexit 1\n"
    )


class EnvSectionTest(unittest.TestCase):
    def test_extracts_only_the_environment_section(self):
        section = runner.env_section(
            SUITE_SCRIPT_TEMPLATE.format(allocator="address", conditional="yes")
        )

        self.assertIn("XLA_PYTHON_CLIENT_ALLOCATOR=address", section)
        # Nothing that installs or runs tests may end up in there.
        self.assertNotIn("install wheels", section)
        self.assertNotIn("exit 1", section)

    def test_no_section_is_reported_as_empty(self):
        self.assertEqual(runner.env_section("#!/bin/bash\nexit 0\n"), "")

    def test_section_without_an_end_marker(self):
        script = f"{runner.ENV_SECTION_START}\nexport A=1\n"

        self.assertEqual(runner.env_section(script), "")


@unittest.skipIf(platform.system() == "Windows", "the suite script needs bash")
class SuiteEnvironmentTest(unittest.TestCase):
    """The suite script is the source of truth, so these read it rather than
    restating what it sets."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.jax_dir = Path(self.tmp.name) / "jax"
        self.addCleanup(self.tmp.cleanup)

    def suite_env(self, **kwargs) -> dict:
        write_suite_script(self.jax_dir, **kwargs)
        return runner.suite_environment(self.jax_dir, dict(os.environ))

    def test_reads_the_allocator_the_script_sets(self):
        self.assertEqual(
            self.suite_env(allocator="platform")["XLA_PYTHON_CLIENT_ALLOCATOR"],
            "platform",
        )
        self.assertEqual(
            self.suite_env(allocator="address")["XLA_PYTHON_CLIENT_ALLOCATOR"],
            "address",
        )

    def test_evaluates_conditionals_rather_than_scraping_them(self):
        # 0.10.1 and 0.10.2 pick their XLA flags at runtime, so reading the text
        # would take whichever branch came last.
        self.assertEqual(
            self.suite_env(conditional="yes")["XLA_FLAGS"],
            "--xla_gpu_enable_cublaslt=false",
        )
        self.assertEqual(
            self.suite_env(conditional="")["XLA_FLAGS"],
            "--xla_gpu_enable_command_buffer=",
        )

    def test_default_env_is_sourced_for_the_variables_it_defines(self):
        self.assertEqual(self.suite_env()["JAX_ENABLE_X64"], "0")

    def test_skips_the_suite_bookkeeping(self):
        env = self.suite_env()

        # The section's own scratch variables, and the xdist worker count, which
        # is for the parallel run and would contradict the serial retry.
        self.assertNotIn("gpu_count", env)
        self.assertNotIn("JAX_ENABLE_ROCM_XDIST", env)

    def test_reports_only_what_the_section_changed(self):
        env = self.suite_env()

        # Dumping the environment on either side of the section keeps everything
        # the shell brought with it out of the result.
        self.assertNotIn("PWD", env)
        self.assertNotIn("SHLVL", env)
        self.assertNotIn("PATH", env)
        self.assertNotIn("JAXCI_ENABLE_X64", env)

    def test_only_the_section_is_evaluated(self):
        # Anything outside the markers would mean installs or tests ran.
        self.assertNotIn("SHOULD_NOT_LEAK", self.suite_env())

    def test_the_retry_pass_inherits_skipping_slow_tests(self):
        self.assertEqual(self.suite_env()["JAX_SKIP_SLOW_TESTS"], "true")

    def test_the_environment_file_is_read_when_the_checkout_has_one(self):
        # Newer ROCm/jax keeps the environment in its own file rather than in a
        # marked section of the suite script.
        write_split_suite_script(self.jax_dir, allocator="platform")

        env = runner.suite_environment(self.jax_dir, dict(os.environ))

        self.assertEqual(env["XLA_PYTHON_CLIENT_ALLOCATOR"], "platform")
        self.assertNotIn("SHOULD_NOT_LEAK", env)

    def test_a_renamed_section_is_fatal(self):
        # Carrying on would run the retry pass, which decides the result, under an
        # environment that does not match the run it is checking.
        self.jax_dir.joinpath("ci").mkdir(parents=True)
        (self.jax_dir / runner.RELATIVE_SUITE_SCRIPT).write_text(
            "#!/bin/bash\nexit 0\n"
        )

        with self.assertRaises(runner.SuiteEnvironmentError) as caught:
            runner.suite_environment(self.jax_dir, dict(os.environ))

        # The message has to say what to fix, since the scripts live upstream.
        self.assertIn(runner.ENV_SECTION_START, str(caught.exception))
        self.assertIn("ENV_SECTION_START", str(caught.exception))
        self.assertIn(
            os.fspath(runner.RELATIVE_SUITE_ENV_SCRIPT), str(caught.exception)
        )

    def test_a_section_that_fails_is_fatal(self):
        # Its device queries feed the values it exports, so a section that
        # ends badly exported something computed from a command that did not
        # work, and the retry pass would run under that.
        write_suite_script(self.jax_dir)
        script = (self.jax_dir / runner.RELATIVE_SUITE_SCRIPT).read_text()
        (self.jax_dir / runner.RELATIVE_SUITE_SCRIPT).write_text(
            script.replace(runner.ENV_SECTION_END, "false\n# Run tests")
        )

        with self.assertRaises(runner.SuiteEnvironmentError) as caught:
            runner.suite_environment(self.jax_dir, dict(os.environ))

        self.assertIn("exited 1", str(caught.exception))

    def test_a_missing_default_env_is_fatal(self):
        write_suite_script(self.jax_dir)
        (self.jax_dir / runner.RELATIVE_SUITE_ENV_FILE).unlink()

        with self.assertRaises(runner.SuiteEnvironmentError):
            runner.suite_environment(self.jax_dir, dict(os.environ))


class TestEnvironmentTest(unittest.TestCase):
    def test_suite_values_are_carried_through(self):
        env = runner.test_environment(
            {"XLA_FLAGS": "--from-the-script"}, "0.11.0", GFX94_FAMILY, 2
        )

        self.assertEqual(env["XLA_FLAGS"], "--from-the-script")

    def test_therock_workarounds_are_added(self):
        env = runner.test_environment({}, "0.11.0", GFX94_FAMILY, 2)

        self.assertEqual(env["ROCPROFILER_QUEUE_INTERPOSITION"], "0")
        self.assertEqual(env["DEBUG_HIP_DYNAMIC_QUEUES"], "0")
        self.assertEqual(env["HSA_NO_SCRATCH_RECLAIM"], "1")

    def test_the_workarounds_win_over_the_script(self):
        # They exist because the runtime needs them here, whatever the suite says.
        env = runner.test_environment(
            {"HSA_NO_SCRATCH_RECLAIM": "0"}, "0.11.0", GFX94_FAMILY, 2
        )

        self.assertEqual(env["HSA_NO_SCRATCH_RECLAIM"], "1")

    def test_addopts_are_ours(self):
        env = runner.test_environment(
            {"PYTEST_ADDOPTS": "--from-the-script"}, "0.10.2", GFX94_FAMILY, 2
        )

        self.assertIn("--reruns", env["PYTEST_ADDOPTS"])
        self.assertIn("not conv", env["PYTEST_ADDOPTS"])

    def test_nothing_to_add_leaves_addopts_alone(self):
        # Wiping them would drop whatever the caller had set.
        env = runner.test_environment(
            {"PYTEST_ADDOPTS": "--from-the-caller"}, "", "", 0
        )

        self.assertEqual(env["PYTEST_ADDOPTS"], "--from-the-caller")


class RetryCommandTest(unittest.TestCase):
    def test_runs_serially_in_this_interpreter(self):
        nodeids = ["tests/a_test.py::test_one", "tests/b_test.py::test_two[a b]"]

        command = runner.retry_command(nodeids)

        self.assertEqual(command[0], sys.executable)
        # -n 0 disables xdist: the point of the pass is a fresh single process.
        self.assertIn("-n", command)
        self.assertEqual(command[command.index("-n") + 1], "0")
        # Each nodeid stays one argument, spaces and all.
        self.assertEqual(command[-2:], nodeids)


class RequestedCpusTest(unittest.TestCase):
    """Reading the pod's CPU allocation off the CI runner's variable.

    Kubernetes has two spellings and the runners have been seen using both, so
    these pin down what each one means rather than trusting the platform docs,
    which say millicores where the gfx942 runners say 25.
    """

    def cpus(self, value, available=192):
        return runner.requested_cpus({runner.CPU_REQUEST_VAR: value}, available)

    def test_whole_cores(self):
        self.assertEqual(self.cpus("25"), 25)

    def test_millicores(self):
        self.assertEqual(self.cpus("25000m"), 25)

    def test_a_bare_number_too_big_for_the_machine_is_millicores(self):
        self.assertEqual(self.cpus("25000"), 25)

    def test_fractional_cores_round_down(self):
        self.assertEqual(self.cpus("7.5"), 7)

    def test_less_than_a_core_still_gets_one(self):
        self.assertEqual(self.cpus("500m"), 1)

    def test_never_more_than_the_machine_has(self):
        self.assertEqual(self.cpus("64", available=8), 8)

    def test_unset_and_unusable_values(self):
        for value in ["", "   ", "garbage", "0"]:
            with self.subTest(value=value):
                self.assertIsNone(self.cpus(value))

    def test_a_missing_variable(self):
        self.assertIsNone(runner.requested_cpus({}, 192))


class OpenblasThreadsTest(unittest.TestCase):
    def test_the_per_worker_share(self):
        # 16 workers is what the suite starts, so a 96-core pod gets 6 each.
        self.assertEqual(runner.openblas_threads(96), 6)

    def test_a_small_pod_still_gets_a_thread(self):
        # 25 cores over 16 workers rounds to nothing, and zero would mean
        # "size it yourself", which is what we are trying to stop.
        self.assertEqual(runner.openblas_threads(25), 1)

    def test_it_lands_in_the_environment(self):
        env = runner.test_environment({}, "0.11.0", GFX94_FAMILY, 2, cpus=96)

        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "6")

    def test_an_unknown_allocation_leaves_it_alone(self):
        env = runner.test_environment({}, "0.11.0", GFX94_FAMILY, 2, cpus=None)

        self.assertNotIn("OPENBLAS_NUM_THREADS", env)


@unittest.skipUnless(hasattr(os, "sched_setaffinity"), "pinning needs Linux")
class LimitToCpusTest(unittest.TestCase):
    def setUp(self):
        self.original = os.sched_getaffinity(0)
        self.addCleanup(os.sched_setaffinity, 0, self.original)

    def test_pins_to_that_many_of_the_cpus_it_had(self):
        if len(self.original) < 2:
            self.skipTest("needs more than one cpu to narrow down to one")

        chosen = runner.limit_to_cpus(1)

        self.assertEqual(len(chosen), 1)
        self.assertEqual(os.sched_getaffinity(0), set(chosen))
        # And what it pinned to has to be a cpu it was allowed to use.
        self.assertTrue(set(chosen) <= self.original)


class CollectReportsTest(unittest.TestCase):
    """Which reports may stand in for the suite result.

    The retry pass decides the run, so it may only be handed the failures of a
    pytest session that reached the end of what it was asked to run.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.jax_dir = Path(self.tmp.name) / "jax"
        (self.jax_dir / "logs").mkdir(parents=True)
        self.addCleanup(self.tmp.cleanup)

    def write(self, name: str, exitcode, nodeids: list[str] | None = None):
        report = {"tests": [{"nodeid": n, "outcome": "failed"} for n in nodeids or []]}
        if exitcode is not None:
            report["exitcode"] = exitcode
        (self.jax_dir / "logs" / name).write_text(json.dumps(report))

    def collect(self) -> runner.SuiteReports:
        return runner.collect_reports(self.jax_dir)

    def test_a_finished_session_with_failures_decides(self):
        self.write("pytest_results_a.json", 1, ["tests/a_test.py::test_one"])

        self.assertEqual(
            self.collect(), runner.SuiteReports(["tests/a_test.py::test_one"], True)
        )

    def test_failures_are_gathered_across_reports(self):
        self.write("pytest_results_a.json", 1, ["tests/a_test.py::test_one"])
        self.write("pytest_results_b.json", 1, ["tests/b_test.py::test_two"])

        self.assertCountEqual(
            self.collect().failed,
            ["tests/a_test.py::test_one", "tests/b_test.py::test_two"],
        )

    def test_a_session_that_stopped_early_does_not(self):
        # 2 interrupted, 3 internal error, 4 usage error, 5 nothing collected.
        for exitcode in (2, 3, 4, 5):
            with self.subTest(exitcode=exitcode):
                self.write("pytest_results_a.json", exitcode, ["tests/a_test.py::t"])

                self.assertFalse(self.collect().complete)

    def test_every_session_has_to_have_finished(self):
        self.write("pytest_results_a.json", 1, ["tests/a_test.py::test_one"])
        self.write("pytest_results_b.json", 2)

        self.assertFalse(self.collect().complete)

    def test_reports_of_only_passes_do_not_speak_for_a_failed_suite(self):
        # Nothing ordinary failed, so whatever failed the suite happened outside
        # the reports and no retry can clear it.
        self.write("pytest_results_a.json", 0)
        self.write("pytest_results_b.json", 0)

        self.assertFalse(self.collect().complete)

    def test_a_report_without_an_exitcode_does_not(self):
        self.write("pytest_results_a.json", None, ["tests/a_test.py::test_one"])

        self.assertFalse(self.collect().complete)

    def test_a_malformed_report_does_not(self):
        (self.jax_dir / "logs" / "pytest_results_a.json").write_text("{truncated")

        self.assertFalse(self.collect().complete)

    def test_no_reports_at_all_do_not(self):
        self.assertEqual(self.collect(), runner.SuiteReports([], False))


class RemoveReportsTest(unittest.TestCase):
    def test_only_the_reports_go(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            (logs / "pytest_results_single.json").write_text("{}")
            (logs / "pytest.log").write_text("kept")

            runner.remove_reports(Path(tmp))

            self.assertFalse((logs / "pytest_results_single.json").exists())
            self.assertTrue((logs / "pytest.log").exists())

    def test_nothing_to_remove_is_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner.remove_reports(Path(tmp))


class CmdArgumentsTest(unittest.TestCase):
    def test_defaults(self):
        args = runner.cmd_arguments([])

        self.assertEqual(args.jax_dir, Path("jax"))
        self.assertEqual(args.in_process_reruns, 2)
        self.assertTrue(args.fresh_process_retry)
        self.assertEqual(args.max_retry_tests, 40)
        self.assertFalse(args.debug)

    def test_fresh_process_retry_can_be_turned_off(self):
        args = runner.cmd_arguments(["--no-fresh-process-retry"])

        self.assertFalse(args.fresh_process_retry)

    def test_no_retries_turns_off_both_layers(self):
        args = runner.cmd_arguments(["--no-retries"])

        self.assertEqual(args.in_process_reruns, 0)
        self.assertFalse(args.fresh_process_retry)

    def test_version_takes_precedence_over_ref(self):
        args = runner.cmd_arguments(
            ["--jax-version", "0.10.1", "--jax-ref", "rocm-jaxlib-v0.11.0"]
        )

        self.assertEqual(
            args.jax_version or runner.jax_version_from_ref(args.jax_ref), "0.10.1"
        )


@unittest.skipIf(platform.system() == "Windows", "the suite script needs bash")
class OrchestrationTest(unittest.TestCase):
    """Covers what the suite result leads to, with the commands stubbed out."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.jax_dir = Path(self.tmp.name) / "jax"
        write_suite_script(self.jax_dir)
        (self.jax_dir / "logs").mkdir()
        self.addCleanup(self.tmp.cleanup)

        self.commands = []
        self.returncodes = []
        self.suite_reports = []
        patcher = mock.patch.object(runner, "run_command", self.fake_run_command)
        patcher.start()
        self.addCleanup(patcher.stop)

        # main() pins itself to the pod's cpus, which would leave this test
        # process pinned for good on a machine that has the variable set.
        self.pinned_to = []
        pinning = mock.patch.object(
            runner, "limit_to_cpus", lambda cores: self.pinned_to.append(cores) or [0]
        )
        pinning.start()
        self.addCleanup(pinning.stop)

        # An allocation is never more than the node has, so stand in for a node
        # the size of the ones the suite runs on. Read for real, a four-core
        # runner would hold a 96-core pod to four.
        node = mock.patch.object(runner, "available_cpus", lambda: 192)
        node.start()
        self.addCleanup(node.stop)

    def fake_run_command(self, args, cwd, env):
        self.commands.append(args)
        if args == ["bash", os.fspath(runner.RELATIVE_SUITE_SCRIPT)]:
            for write in self.suite_reports:
                write()
        return self.returncodes.pop(0)

    def suite_writes_report(self, name: str, nodeids: list[str], exitcode: int = 1):
        """Queues a report for the fake suite to leave behind, as pytest would.

        Writing it up front would not survive: the run clears the reports of an
        earlier one before starting.
        """
        self.suite_reports.append(lambda: self.write_report(name, nodeids, exitcode))

    def write_report(self, name: str, nodeids: list[str], exitcode: int = 1):
        (self.jax_dir / "logs" / name).write_text(
            json.dumps(
                {
                    "exitcode": exitcode,
                    "tests": [
                        {"nodeid": nodeid, "outcome": "failed"} for nodeid in nodeids
                    ],
                }
            )
        )

    def run_main(self, extra: list[str] | None = None) -> int:
        argv = ["--jax-dir", os.fspath(self.jax_dir), "--jax-version", "0.11.0"]
        return runner.main(argv + (extra or []))

    def test_a_passing_suite_is_not_retried(self):
        self.returncodes = [0, 0]

        self.assertEqual(self.run_main(), 0)
        self.assertEqual(len(self.commands), 2)

    def test_the_suite_is_held_to_the_pods_cpus(self):
        self.returncodes = [0, 0]
        environments = []
        with mock.patch.dict(os.environ, {runner.CPU_REQUEST_VAR: "96"}):
            with mock.patch.object(
                runner,
                "run_command",
                lambda args, cwd, env: (
                    environments.append(env) or self.fake_run_command(args, cwd, env)
                ),
            ):
                self.run_main()

        self.assertEqual(self.pinned_to, [96])
        self.assertEqual(environments[1]["OPENBLAS_NUM_THREADS"], "6")

    def test_an_unknown_allocation_leaves_the_run_alone(self):
        # Off a CI runner there is nothing to read, and sizing pools to a
        # fraction of a machine nobody else is using would only slow it down.
        self.returncodes = [0, 0]
        with mock.patch.dict(os.environ, {runner.CPU_REQUEST_VAR: ""}):
            self.run_main()

        self.assertEqual(self.pinned_to, [])

    def test_cpus_can_be_given_on_the_command_line(self):
        self.returncodes = [0, 0]
        with mock.patch.dict(os.environ, {runner.CPU_REQUEST_VAR: ""}):
            self.run_main(["--cpus", "32"])

        self.assertEqual(self.pinned_to, [32])

    def test_the_suite_runs_under_the_environment_read_from_it(self):
        self.returncodes = [0, 0]
        environments = []
        with mock.patch.object(
            runner,
            "run_command",
            lambda args, cwd, env: (
                environments.append(env) or self.fake_run_command(args, cwd, env)
            ),
        ):
            self.run_main()

        self.assertEqual(environments[1]["XLA_PYTHON_CLIENT_ALLOCATOR"], "address")
        self.assertEqual(environments[1]["HSA_NO_SCRATCH_RECLAIM"], "1")

    def test_an_unreadable_environment_stops_before_the_suite(self):
        (self.jax_dir / runner.RELATIVE_SUITE_SCRIPT).write_text(
            "#!/bin/bash\nexit 1\n"
        )
        self.returncodes = []

        self.assertEqual(self.run_main(), 1)
        self.assertEqual(self.commands, [])

    def test_the_preflight_check_runs_before_the_suite(self):
        self.returncodes = [0, 0]

        self.run_main()

        self.assertEqual(self.commands[0], runner.preflight_command())
        self.assertEqual(
            self.commands[1], ["bash", os.fspath(runner.RELATIVE_SUITE_SCRIPT)]
        )
        # The suite script fails without this directory.
        self.assertTrue((self.jax_dir / "dist").is_dir())

    def test_a_failing_preflight_check_skips_the_suite(self):
        # A stack that sees no device would report every test as an error.
        self.returncodes = [1]

        self.assertEqual(self.run_main(), 1)
        self.assertEqual(len(self.commands), 1)

    def test_reported_failures_are_retried_in_a_fresh_process(self):
        self.suite_writes_report(
            "pytest_results_single.json", ["tests/a_test.py::test_one"]
        )
        self.suite_writes_report(
            "pytest_results_multi.json", ["tests/b_test.py::test_two"]
        )
        self.returncodes = [0, 1, 0]

        # The retry decides the result: it passed, so the job passes.
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(len(self.commands), 3)
        # Every report the suite wrote is retried, in whatever order they glob.
        self.assertCountEqual(
            self.commands[2][-2:],
            ["tests/a_test.py::test_one", "tests/b_test.py::test_two"],
        )

    def test_a_failing_retry_fails_the_job(self):
        self.suite_writes_report(
            "pytest_results_single.json", ["tests/a_test.py::test_one"]
        )
        self.returncodes = [0, 1, 1]

        self.assertEqual(self.run_main(), 1)

    def test_a_passing_session_alongside_a_failing_one_still_decides(self):
        # The suite makes one pytest run per configuration, and only the one
        # that failed has anything to retry.
        self.suite_writes_report("pytest_results_multi.json", [], exitcode=0)
        self.suite_writes_report(
            "pytest_results_single.json", ["tests/a_test.py::test_one"], exitcode=1
        )
        self.returncodes = [0, 1, 0]

        self.assertEqual(self.run_main(), 0)
        self.assertEqual(len(self.commands), 3)

    def test_an_interrupted_session_keeps_the_suite_result(self):
        # Exit code 2 is a session pytest stopped, so the tests it never reached
        # are missing from the report and retrying the rest proves nothing.
        self.suite_writes_report(
            "pytest_results_single.json", ["tests/a_test.py::test_one"], exitcode=2
        )
        self.returncodes = [0, 1]

        self.assertEqual(self.run_main(), 1)
        self.assertEqual(len(self.commands), 2)

    def test_one_unfinished_session_holds_back_the_whole_retry(self):
        self.suite_writes_report(
            "pytest_results_single.json", ["tests/a_test.py::test_one"], exitcode=1
        )
        self.suite_writes_report("pytest_results_multi.json", [], exitcode=3)
        self.returncodes = [0, 1]

        self.assertEqual(self.run_main(), 1)
        self.assertEqual(len(self.commands), 2)

    def suite_writes_raw_report(self, name: str, text: str):
        """Queues a report the suite writes verbatim, sound or not."""
        self.suite_reports.append(
            lambda: (self.jax_dir / "logs" / name).write_text(text)
        )

    def test_a_report_without_an_exitcode_keeps_the_suite_result(self):
        # The retryable-looking failures of the sound report are not enough: the
        # session that wrote the other one may have stopped anywhere.
        self.suite_writes_report(
            "pytest_results_a.json", ["tests/a_test.py::test_one"], exitcode=1
        )
        self.suite_writes_raw_report(
            "pytest_results_b.json",
            json.dumps({"tests": [{"nodeid": "tests/b.py::t", "outcome": "failed"}]}),
        )
        self.returncodes = [0, 1]

        self.assertEqual(self.run_main(), 1)
        self.assertEqual(len(self.commands), 2)

    def test_a_malformed_report_keeps_the_suite_result(self):
        self.suite_writes_report(
            "pytest_results_a.json", ["tests/a_test.py::test_one"], exitcode=1
        )
        self.suite_writes_raw_report("pytest_results_b.json", "{truncated")
        self.returncodes = [0, 1]

        self.assertEqual(self.run_main(), 1)
        self.assertEqual(len(self.commands), 2)

    def test_reports_of_an_earlier_run_are_removed_first(self):
        # Otherwise a local rerun retries the failures of the run before it and
        # passes on results this suite never produced.
        stale = self.jax_dir / "logs" / "pytest_results_single.json"
        self.write_report("pytest_results_single.json", ["tests/a_test.py::test_one"])
        self.returncodes = [0, 1]

        self.assertEqual(self.run_main(), 1)
        self.assertFalse(stale.exists())
        self.assertEqual(len(self.commands), 2)

    def test_nothing_to_retry_keeps_the_suite_result(self):
        # A suite that fails without reporting a failed test, e.g. it died
        # during collection, has nothing for the retry pass to work with.
        self.returncodes = [0, 1]

        self.assertEqual(self.run_main(), 1)
        self.assertEqual(len(self.commands), 2)

    def test_too_many_failures_skips_the_retry(self):
        self.suite_writes_report(
            "pytest_results_single.json",
            [f"tests/a_test.py::test_{i}" for i in range(5)],
        )
        self.returncodes = [0, 1]

        self.assertEqual(self.run_main(["--max-retry-tests", "4"]), 1)
        self.assertEqual(len(self.commands), 2)

    def test_retry_can_be_turned_off(self):
        self.suite_writes_report(
            "pytest_results_single.json", ["tests/a_test.py::test_one"]
        )
        self.returncodes = [0, 1]

        self.assertEqual(self.run_main(["--no-fresh-process-retry"]), 1)
        self.assertEqual(len(self.commands), 2)

    def test_no_retries_runs_only_the_suite(self):
        self.suite_writes_report(
            "pytest_results_single.json", ["tests/a_test.py::test_one"]
        )
        self.returncodes = [0, 1]

        self.assertEqual(self.run_main(["--no-retries"]), 1)
        self.assertEqual(len(self.commands), 2)

    def test_missing_checkout_is_an_error(self):
        self.returncodes = []

        self.assertEqual(
            runner.main(["--jax-dir", os.fspath(self.jax_dir / "absent")]), 1
        )
        self.assertEqual(self.commands, [])


class ReportedEnvTest(unittest.TestCase):
    def test_visibility_variables_are_reported(self):
        # Which physical GPU the job got, for reading a failure after the fact.
        self.assertEqual(
            runner.GPU_VISIBILITY_ENV_VARS,
            ["ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"],
        )


class UnsetVarsTest(unittest.TestCase):
    def test_plugin_build_paths_are_dropped(self):
        # Wheels carry their own bitcode and linker, so a value left over from a
        # build points at paths that do not exist in a test job.
        for name in [
            "ROCM_ROOT",
            "HIP_DEVICE_LIB_PATH",
            "JAX_ROCM_PLUGIN_INTERNAL_BITCODE_PATH",
            "JAX_ROCM_PLUGIN_INTERNAL_LLD_PATH",
        ]:
            self.assertIn(name, runner.UNSET_VARS)


if __name__ == "__main__":
    unittest.main()
