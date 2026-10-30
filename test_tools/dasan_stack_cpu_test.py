# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import dasan_stack_cpu as cpu


class StackCpuTest(unittest.TestCase):
    def test_only_success_and_no_device_are_accepted(self):
        self.assertFalse(
            cpu.check_result("DASAN_STACK_CPU hip_status=0 devices=2\n")["no_gpu"]
        )
        self.assertTrue(
            cpu.check_result("DASAN_STACK_CPU hip_status=100 devices=0\n")["no_gpu"]
        )
        for result in (
            "DASAN_STACK_CPU hip_status=999 devices=0\n",
            "DASAN_STACK_CPU hip_status=0 devices=-1\n",
            "",
            "DASAN_STACK_CPU hip_status=100 devices=0\n" * 2,
        ):
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                cpu.check_result(result)

    def test_shared_runtime_selected_and_linked_with_both_dsos(self):
        with tempfile.TemporaryDirectory() as temporary:
            rocm = Path(temporary) / "rocm"
            runtime = rocm / "lib/llvm/lib/clang/custom/lib/host/libclang_rt.dasan.so"
            runtime.parent.mkdir(parents=True)
            runtime.touch()
            output = Path(temporary) / "logs"
            with patch.object(
                cpu,
                "command",
                side_effect=[
                    str(runtime),
                    "",
                    "DASAN_STACK_CPU hip_status=100 devices=0\n",
                ],
            ) as commands:
                result = cpu.validate(rocm, output)
            self.assertEqual(result["state"], "passed")
            argv = commands.call_args_list[1].args[0]
            self.assertIn("-shared-libsan", argv)
            self.assertIn("-fsanitize=daddress", argv)
            self.assertIn("-lamdhip64", argv)
            self.assertIn("-lhsa-runtime64", argv)
            self.assertTrue(
                commands.call_args_list[2]
                .args[1]["LD_LIBRARY_PATH"]
                .startswith(str(runtime.parent) + os.pathsep)
            )

    def test_runtime_must_belong_to_selected_installation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external = root / "external/libclang_rt.dasan.so"
            external.parent.mkdir()
            external.touch()
            with patch.object(
                cpu, "command", return_value=str(external)
            ), self.assertRaisesRegex(RuntimeError, "outside selected"):
                cpu.validate(root / "rocm", root / "logs")
            summary = json.loads((root / "logs/summary.json").read_text())
            self.assertEqual(summary["state"], "failed")

    def test_process_crash_is_never_accepted_as_no_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "execute.log"
            with patch.object(
                cpu.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(["test"], -11, ""),
            ), self.assertRaisesRegex(RuntimeError, "exited -11"):
                cpu.command(["test"], {}, log)
            self.assertTrue(log.exists())


if __name__ == "__main__":
    unittest.main()
