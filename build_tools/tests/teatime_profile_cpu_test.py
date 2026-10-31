# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

TEST_PATH = Path(__file__).resolve()
SCRIPT = (
    TEST_PATH.parents[1] / "teatime.py"
    if TEST_PATH.parent.name == "tests"
    else TEST_PATH.with_name("teatime.py")
)
spec = importlib.util.spec_from_file_location("teatime_cpu_candidate", SCRIPT)
teatime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(teatime)


def records(path):
    return [
        json.loads(line.split("\t", 1)[1])
        for line in path.read_text().splitlines()
        if line.startswith("CPU_PROFILE\t")
    ]


class TeatimeCpuProfileTest(unittest.TestCase):
    def invoke(self, root, command=None, enabled=True, input_text=None):
        log = Path(root) / "phase.log"
        env = dict(os.environ, TEATIME_PROFILE_CPU="1" if enabled else "0")
        args = [
            sys.executable,
            str(SCRIPT),
            "--log-timestamps",
            "--label",
            'project "build"',
            "--no-interactive",
            str(log),
        ]
        if command is not None:
            args += ["--", *command]
        run = subprocess.run(
            args, input=input_text, capture_output=True, text=True, env=env
        )
        return run, log

    def test_default_off_preserves_existing_log_shape_and_failure_exit(self):
        with tempfile.TemporaryDirectory() as root:
            run, log = self.invoke(
                root,
                [sys.executable, "-c", "print('child output');raise SystemExit(7)"],
                enabled=False,
            )
            self.assertEqual(run.returncode, 7)
            self.assertEqual(records(log), [])
            lines = log.read_text().splitlines()
            self.assertTrue(lines[0].startswith("BEGIN\t"))
            self.assertTrue(lines[1].startswith("EXEC\t"))
            self.assertTrue(lines[-1].startswith("END\t"))
            self.assertTrue(lines[-1].endswith("\t7"))
            self.assertIn("child output", run.stdout)

    def test_pipeline_cannot_report_child_cpu(self):
        with tempfile.TemporaryDirectory() as root:
            run, log = self.invoke(root, input_text="pipeline input\n")
            self.assertEqual(run.returncode, 0)
            values = records(log)
            self.assertEqual(len(values), 1)
            value = values[0]
            self.assertEqual(value["status"], "unavailable_pipeline")
            self.assertIsNone(value["child_pid"])
            self.assertIsNone(value["child_total_s"])
            self.assertIsNone(value["child_user_s"])
            self.assertIsNone(value["child_system_s"])

    def test_subprocess_failure_keeps_exit_and_structured_record(self):
        with tempfile.TemporaryDirectory() as root:
            run, log = self.invoke(root, [sys.executable, "-c", "raise SystemExit(9)"])
            self.assertEqual(run.returncode, 9)
            values = records(log)
            self.assertEqual(len(values), 1)
            value = values[0]
            self.assertEqual(value["exit_code"], 9)
            self.assertEqual(value["label"], 'project "build"')
            self.assertEqual(Path(value["cwd"]), Path.cwd())
            self.assertGreaterEqual(value["phase_wall_s"], 0)
            self.assertIsInstance(value["child_pid"], int)
            self.assertTrue(log.read_text().splitlines()[-1].endswith("\t9"))
            if os.name == "posix":
                self.assertEqual(value["status"], "measured_posix_waited_children")
                self.assertGreaterEqual(value["child_total_s"], 0)
            else:
                self.assertEqual(value["status"], "unavailable_non_posix")
                for field in ["child_user_s", "child_system_s", "child_total_s"]:
                    self.assertIsNone(value[field])

    @unittest.skipUnless(
        os.name == "posix", "requires real POSIX kernel child accounting"
    )
    def test_kernel_child_total_includes_waited_descendant_and_excludes_parent_cpu(
        self,
    ):
        import resource

        nested = (
            "import time;t=time.process_time();\nwhile time.process_time()-t<.15:pass"
        )
        child = (
            "import subprocess,sys,time;print('begin',flush=True);"
            f"subprocess.run([sys.executable,'-c',{nested!r}],check=True);"
            "t=time.process_time();\nwhile time.process_time()-t<.2:pass"
        )
        with tempfile.TemporaryDirectory() as root, patch.dict(
            os.environ, {"TEATIME_PROFILE_CPU": "1"}
        ):
            log = Path(root) / "phase.log"
            args = argparse.Namespace(
                file=log, label="nested build", interactive=False, log_timestamps=True
            )
            sink = teatime.OutputSink(args)
            original_writeline = sink.writeline

            def parent_work(line):
                start = time.process_time()
                while time.process_time() - start < 0.55:
                    pass
                original_writeline(line)

            sink.writeline = parent_work
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            parent_before = time.process_time()
            sink.start()
            try:
                with self.assertRaises(SystemExit) as caught:
                    teatime.run(args, [sys.executable, "-c", child], sink)
                self.assertEqual(caught.exception.code, 0)
            finally:
                sink.finish(0)
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            value = records(log)[0]
            expected_user = after.ru_utime - before.ru_utime
            expected_system = after.ru_stime - before.ru_stime
            self.assertAlmostEqual(value["child_user_s"], expected_user, places=7)
            self.assertAlmostEqual(value["child_system_s"], expected_system, places=7)
            self.assertAlmostEqual(
                value["child_total_s"], expected_user + expected_system, places=7
            )
            self.assertGreater(value["child_total_s"], 0.3)
            self.assertGreaterEqual(time.process_time() - parent_before, 0.55)


if __name__ == "__main__":
    unittest.main(verbosity=2)
