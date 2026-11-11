#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise diagnostic preservation with an actual, GPU-free child runner."""

from contextlib import contextmanager, nullcontext, redirect_stdout, redirect_stderr
import io
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import re
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_debug_agent_tests import (
    MEMORY_PATTERNS,
    SIGQUIT_PATTERNS,
    adapt_installed_suite,
    adapted_harness,
    main,
    valid_stopped_pc,
)


# Reduced verbatim wave/disassembly records from TheRock test43017881751.
# Only unrelated register lanes and source-line listings are omitted.
MEMORY_DUMP = """wave_4: pc=0x774e08c0ce80 (kernel_code_entry=0x774e08c0cf00 <vector_add_memory_fault(int*, int*, int*)>, kernargs=0x774e04c00100) (stopped, reason: MEMORY_VIOLATION)
 s0: 08c0ce80
 v0: [0] 00000000
 pc: 0000774e08c0ce80          exec: ffffffffffffffff
 0x0000: 22222222 11111111 22222222 11111111
Disassembly for function __csan_func_exit:
    loaded at: [0x774e08c08000-0x774e08c0fd90]
 => 0x774e08c0ce80 <+0>:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
    0x774e08c0ce84 <+4>:    s_setpc_b64 s[30:31]
End of disassembly.
"""
SIGQUIT_DUMP = """wave_1: pc=0x767c45a8e080 (kernel_code_entry=0x767c45a8e500 <sigquit_resume_kern(int*, int const*)>, kernargs=0x767c34200000) (running)
Disassembly for function __csan_read4:
    loaded at: [0x767c45a8c000-0x767c45a90eb8]
 => 0x767c45a8e080 <+0>:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
    0x767c45a8e084 <+4>:    s_mov_b32 s15, s33
End of disassembly.
"""
MODE_ZERO_MEMORY_DUMP = """wave_2: pc=0x71cc5ce23248 (kernel_code_entry=0x71cc5ce24f00 <vector_add_memory_fault(int*, int*, int*)>, kernargs=0x71cc50c00100) (stopped, reason: MEMORY_VIOLATION)
Disassembly for function check_access(void const volatile*, unsigned long, int, unsigned long):
    loaded at: [0x71cc5ce20000-0x71cc5ce27d90]
/__w/TheRock/TheRock/compiler/amd-llvm/compiler-rt/lib/csan/csan_gpu.cpp:
    0x71cc5ce2322c <+2220>:    global_load_dwordx2 v[6:7], v[28:29], off sc0 sc1
    0x71cc5ce23234 <+2228>:    s_waitcnt vmcnt(0)
    0x71cc5ce23238 <+2232>:    s_andn2_saveexec_b64 s[6:7], s[6:7]
    0x71cc5ce23240 <+2240>:    global_load_dword v6, v[28:29], off sc0 sc1
 => 0x71cc5ce23248 <+2248>:    s_waitcnt vmcnt(0)
    0x71cc5ce2324c <+2252>:    v_mov_b32_e32 v7, 0
    0x71cc5ce23250 <+2256>:    s_or_b64 exec, exec, s[6:7]
End of disassembly.
"""


class CsanStoppedPcTest(unittest.TestCase):
    def test_actual_captured_fault_and_sigquit_records(self):
        self.assertTrue(valid_stopped_pc(MEMORY_DUMP, memory_fault=True))
        self.assertTrue(valid_stopped_pc(SIGQUIT_DUMP, memory_fault=False))

    def test_actual_mode_zero_null_read_probe_and_strict_negatives(self):
        self.assertTrue(valid_stopped_pc(MODE_ZERO_MEMORY_DUMP, memory_fault=True))
        replacements = [
            (
                "check_access(void const volatile*",
                "unrelated_access(void const volatile*",
            ),
            ("compiler-rt/lib/csan/csan_gpu.cpp:", "unrelated.cpp:"),
            ("<+2240>", "<+2236>"),
            ("global_load_dword v6", "global_store_dword v6"),
            ("s_waitcnt vmcnt(0)", "s_nop 0"),
            ("v_mov_b32_e32 v7, 0", "v_mov_b32_e32 v7, 1"),
            ("pc=0x71cc5ce23248", "pc=0x71cc5ce2324c"),
        ]
        for old, new in replacements:
            with self.subTest(old=old):
                self.assertFalse(
                    valid_stopped_pc(
                        MODE_ZERO_MEMORY_DUMP.replace(old, new), memory_fault=True
                    )
                )

    def test_wrong_kernel_reason_helper_pc_instruction_and_bounds_fail(self):
        replacements = [
            ("vector_add_memory_fault", "unrelated_kernel"),
            ("MEMORY_VIOLATION", "ASSERT_TRAP"),
            ("__csan_func_exit", "__csan_write4"),
            ("pc=0x774e08c0ce80", "pc=0x774e08c0ce84"),
            ("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)", "s_nop 0"),
            ("s_setpc_b64 s[30:31]", "s_nop 0"),
            ("<+0>", "<-4>"),
            ("0x774e08c0fd90", "0x774e08c0ce80"),
        ]
        for old, new in replacements:
            with self.subTest(old=old):
                self.assertFalse(
                    valid_stopped_pc(MEMORY_DUMP.replace(old, new), memory_fault=True)
                )
        self.assertFalse(
            valid_stopped_pc(
                SIGQUIT_DUMP.replace("__csan_read4", "__csan_write4"),
                memory_fault=False,
            )
        )
        self.assertFalse(
            valid_stopped_pc(
                SIGQUIT_DUMP.replace("(running)", "(stopped, reason: ASSERT_TRAP)"),
                memory_fault=False,
            )
        )

    def test_disassembly_cannot_come_from_another_wave(self):
        header, body = MEMORY_DUMP.split("Disassembly", 1)
        unrelated = "wave_99: pc=0x1 (kernel_code_entry=0x1 <other()>, kernargs=0x2) (running)\n"
        self.assertFalse(
            valid_stopped_pc(
                header + unrelated + "Disassembly" + body, memory_fault=True
            )
        )

    def test_original_kernel_disassembly_requires_kernel_function_base(self):
        dump = MEMORY_DUMP.replace(
            "Disassembly for function __csan_func_exit",
            "Disassembly for function vector_add_memory_fault(int*, int*, int*)",
        ).replace(
            "kernel_code_entry=0x774e08c0cf00", "kernel_code_entry=0x774e08c0ce80"
        )
        self.assertTrue(valid_stopped_pc(dump, memory_fault=True))
        self.assertFalse(
            valid_stopped_pc(dump.replace("<+0>", "<+4>"), memory_fault=True)
        )

    def test_adapter_preserves_first_five_fault_checks_and_registrations(self):
        definitions = [
            {"name": f"unchanged_{i}", "patterns": ["strict-original"]}
            for i in range(16)
        ] + [
            {"name": "test_memory_violation", "patterns": MEMORY_PATTERNS.copy()},
            {"name": "test_sigquit_resumes", "patterns": SIGQUIT_PATTERNS.copy()},
        ]
        original = repr(definitions)
        namespace = {
            "TEST_DEFINITIONS": definitions,
            "logger": lambda: logging.getLogger(__name__),
            "check_errors": lambda checks, out, err: all(
                check.search(err) for check in checks
            ),
            "agent_loaded_by": nullcontext,
        }
        adapt_installed_suite(namespace)
        check = namespace["check_errors"]
        patterns = [re.compile(pattern) for pattern in MEMORY_PATTERNS]
        self.assertTrue(check(patterns, "", MEMORY_DUMP))
        for missing in (
            "MEMORY_VIOLATION",
            "exec: ffffffffffffffff",
            "s0:",
            "v0:",
            "0x0000: 22222222 11111111",
        ):
            with self.subTest(missing=missing):
                self.assertFalse(
                    check(patterns, "", MEMORY_DUMP.replace(missing, "MISSING"))
                )
        self.assertFalse(check([re.compile("strict-original")], "", MEMORY_DUMP))
        self.assertTrue(check([re.compile("strict-original")], "", "strict-original"))
        self.assertEqual(repr(definitions), original)
        self.assertEqual(len(definitions) * 3, 54)

    def test_unrecognized_installed_structure_fails_closed(self):
        with self.assertRaises(ValueError):
            adapted_harness("TEST_DEFINITIONS = []\n")
        namespace = {"TEST_DEFINITIONS": [{"name": "unexpected"}]}
        with self.assertRaises(ValueError):
            adapt_installed_suite(namespace)

    def test_ast_injection_runs_before_all_54_original_cases(self):
        definitions = [{"name": f"original_{i}"} for i in range(16)] + [
            {"name": "test_memory_violation", "patterns": MEMORY_PATTERNS},
            {"name": "test_sigquit_resumes", "patterns": SIGQUIT_PATTERNS},
        ]
        source = (
            "import logging\n"
            "from contextlib import nullcontext as agent_loaded_by\n"
            "def logger(): return logging.getLogger('installed-harness')\n"
            "def check_errors(check_list, out_str, err_str):\n"
            "    return all(check.search(err_str) for check in check_list)\n"
            f"TEST_DEFINITIONS = {definitions!r}\n"
            "executed = []\n"
            "for mode in (None, '1', '0'):\n"
            "    for definition in TEST_DEFINITIONS:\n"
            "        executed.append((mode, definition['name']))\n"
        )
        namespace = {}
        exec(compile(adapted_harness(source), "installed-harness", "exec"), namespace)
        self.assertEqual(len(namespace["executed"]), 54)
        self.assertEqual(namespace["TEST_DEFINITIONS"], definitions)

    def test_preload_cases_retain_prepared_chain_and_restore_environment(self):
        @contextmanager
        def installed_loader(mode):
            with patch.dict(os.environ):
                if mode == "preload":
                    os.environ["LD_PRELOAD"] = os.environ.pop(
                        "HSA_TOOLS_LIB", "actual-agent.so"
                    )
                yield

        definitions = [{"name": f"original_{i}"} for i in range(16)] + [
            {"name": "test_memory_violation", "patterns": MEMORY_PATTERNS},
            {"name": "test_sigquit_resumes", "patterns": SIGQUIT_PATTERNS},
        ]
        namespace = {
            "TEST_DEFINITIONS": definitions,
            "logger": lambda: logging.getLogger(__name__),
            "check_errors": lambda *args: True,
            "agent_loaded_by": installed_loader,
        }
        chain = "actual-csan.so:actual-hsa.so:actual-report-shim.so:caller.so"
        with patch.dict(
            os.environ, {"LD_PRELOAD": chain, "HSA_TOOLS_LIB": "selected-agent.so"}
        ):
            adapt_installed_suite(namespace)
            with namespace["agent_loaded_by"]("preload"):
                self.assertEqual(os.environ["LD_PRELOAD"], chain + ":selected-agent.so")
                self.assertNotIn("HSA_TOOLS_LIB", os.environ)
            self.assertEqual(os.environ["LD_PRELOAD"], chain)
            self.assertEqual(os.environ["HSA_TOOLS_LIB"], "selected-agent.so")
            with namespace["agent_loaded_by"]("tools"):
                self.assertEqual(os.environ["LD_PRELOAD"], chain)
                self.assertEqual(os.environ["HSA_TOOLS_LIB"], "selected-agent.so")


class DebugAgentRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "artifacts"
        self.test_dir = self.artifacts / "tests" / "rocm-debug-agent"
        self.test_dir.mkdir(parents=True)
        self.diagnostics = self.root / "diagnostics"

    def run_fake(self, exit_code):
        (self.test_dir / "run-test.py").write_text(
            "import os, pathlib, sys\n"
            "assert pathlib.Path(sys.argv[1]) == pathlib.Path.cwd()\n"
            "assert os.environ['DEBUG_AGENT_PROBE'] == 'preserved'\n"
            "with open('calls', 'a') as calls: calls.write('called\\n')\n"
            "pathlib.Path('run-test.log').write_text('child memory-violation detail\\n')\n"
            f"sys.exit({exit_code})\n",
            encoding="utf-8",
        )
        output = io.StringIO()
        with patch.dict(
            os.environ, {"DEBUG_AGENT_PROBE": "preserved"}
        ), redirect_stdout(output), patch(
            "run_debug_agent_tests.subprocess.run", wraps=subprocess.run
        ) as run:
            result = main(
                [
                    "--artifacts-dir",
                    str(self.artifacts),
                    "--diagnostics-dir",
                    str(self.diagnostics),
                ]
            )
        child_env = run.call_args.kwargs["env"]
        self.assertEqual(child_env.get("LD_PRELOAD"), os.environ.get("LD_PRELOAD"))
        self.assertTrue(child_env["PATH"].startswith(str(self.artifacts / "bin")))
        self.assertTrue(
            child_env["LD_LIBRARY_PATH"].startswith(str(self.artifacts / "lib"))
        )
        self.assertEqual((self.test_dir / "calls").read_text(), "called\n")
        self.assertEqual(
            (self.diagnostics / "run-test.log").read_text(),
            "child memory-violation detail\n",
        )
        return result, output.getvalue()

    def test_failure_runs_once_preserves_and_echoes_log_and_exit_code(self):
        result, output = self.run_fake(7)
        self.assertEqual(result, 7)
        self.assertIn("child memory-violation detail", output)

    def test_success_preserves_log_without_failure_dump(self):
        result, output = self.run_fake(0)
        self.assertEqual(result, 0)
        self.assertNotIn("child memory-violation detail", output)

    def test_missing_runner_fails_with_actionable_path(self):
        output = io.StringIO()
        with redirect_stderr(output):
            result = main(
                [
                    "--artifacts-dir",
                    str(self.artifacts),
                    "--diagnostics-dir",
                    str(self.diagnostics),
                ]
            )
        self.assertEqual(result, 1)
        self.assertIn(str(self.test_dir / "run-test.py"), output.getvalue())


if __name__ == "__main__":
    unittest.main()
