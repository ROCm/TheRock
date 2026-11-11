#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the installed debug-agent suite once and retain its child diagnostics."""

import argparse
import ast
import contextlib
import os
from pathlib import Path
import shutil
import re
import subprocess
import sys


MEMORY_PATTERNS = [
    r"\(stopped, reason: MEMORY_VIOLATION\)",
    r"exec: (ffffffff)?ffffffff",
    r"s0:",
    r"v0:",
    r"0x0000: 22222222 11111111",
    r"Disassembly for function vector_add_memory_fault\(int\*, int\*, int\*\)",
]
SIGQUIT_PATTERNS = [
    r"<sigquit_resume_kern\(int\*, int const\*\)>",
    r"Disassembly for function sigquit_resume_kern\(int\*, int const\*\)",
]
_WAVE = re.compile(
    r"^wave_\d+: pc=(0x[0-9a-f]+) \(kernel_code_entry=(0x[0-9a-f]+) "
    r"<([^>]+)>, kernargs=[^\n]+\) \(([^\n]+)\)\n(.*?)(?=^wave_\d+:|\Z)",
    re.MULTILINE | re.DOTALL,
)


def valid_stopped_pc(err: str, *, memory_fault: bool) -> bool:
    """Match the actual PC to its own dispatch and symbolized instruction.

    Memory exceptions are asynchronous by default. CSAN's out-of-line exit
    wait may drain the kernel's faulting store; SIGQUIT can stop in a read probe.
    A null access can also fault inside the instrumented-memory read probe.
    Allow only the helper functions/instruction sequences evidenced in CI.
    """
    kernel = (
        "vector_add_memory_fault(int*, int*, int*)"
        if memory_fault
        else "sigquit_resume_kern(int*, int const*)"
    )
    helper = "__csan_func_exit" if memory_fault else "__csan_read4"
    matching = 0
    for wave in _WAVE.finditer(err):
        pc, entry, actual_kernel, state, body = wave.groups()
        if actual_kernel != kernel:
            continue
        matching += 1
        if state != (
            "stopped, reason: MEMORY_VIOLATION" if memory_fault else "running"
        ):
            return False
        disassembly = re.search(
            r"Disassembly for function ([^\n:]+):\n(.*?)End of disassembly\.",
            body,
            re.DOTALL,
        )
        if not disassembly:
            return False
        function, instructions = disassembly.groups()
        bounds = re.search(r"loaded at: \[(0x[0-9a-f]+)-(0x[0-9a-f]+)\]", instructions)
        marked = re.findall(
            r"^\s*=> (0x[0-9a-f]+) <([+-]\d+)>:\s+([^\n]+)",
            instructions,
            re.MULTILINE,
        )
        if not bounds or len(marked) != 1:
            return False
        address, offset, instruction = marked[0]
        pc_value, entry_value = int(pc, 16), int(entry, 16)
        lower, upper = (int(value, 16) for value in bounds.groups())
        if int(address, 16) != pc_value or not (
            lower <= pc_value < upper and lower <= entry_value < upper
        ):
            return False
        offset_value = int(offset)
        function_base = pc_value - offset_value
        if not lower <= function_base < upper:
            return False
        # Every displayed instruction must agree on the symbol's base address;
        # a matching arrow from a different function/window is insufficient.
        for shown_address, shown_offset in re.findall(
            r"^\s*(?:=> )?(0x[0-9a-f]+) <([+-]\d+)>:", instructions, re.MULTILINE
        ):
            if int(shown_address, 16) - int(shown_offset) != function_base:
                return False
        if function == kernel:
            if offset_value < 0 or pc_value - offset_value != entry_value:
                return False
            if not re.match(
                r"(?:s_|v_|global_|flat_|scratch_|buffer_|ds_)[a-z0-9_]+\b", instruction
            ):
                return False
        elif function == helper:
            if (
                offset_value != 0
                or instruction.strip() != "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)"
            ):
                return False
            # The empty exit helper consists of the wait followed by its return.
            if memory_fault and not re.search(
                rf"^\s*{hex(pc_value + 4)} <\+4>:\s+s_setpc_b64 s\[30:31\]\s*$",
                instructions,
                re.MULTILINE,
            ):
                return False
        elif (
            memory_fault
            and function
            == "check_access(void const volatile*, unsigned long, int, unsigned long)"
        ):
            if offset_value <= 0 or instruction.strip() != "s_waitcnt vmcnt(0)":
                return False
            if "compiler-rt/lib/csan/csan_gpu.cpp:" not in instructions:
                return False
            # The actual size4 CSAN probe reads the deliberately null address.
            # Require the neighboring load/result instructions, not any wait
            # somewhere in an arbitrary function called check_access.
            if not re.search(
                rf"^\s*{hex(pc_value - 8)} <\+{offset_value - 8}>:\s+global_load_dword v\d+, v\[\d+:\d+\], off sc0 sc1\s*$",
                instructions,
                re.MULTILINE,
            ) or not re.search(
                rf"^\s*{hex(pc_value + 4)} <\+{offset_value + 4}>:\s+v_mov_b32_e32 v\d+, 0\s*$",
                instructions,
                re.MULTILINE,
            ):
                return False
        else:
            return False
    return matching > 0


def adapt_installed_suite(namespace):
    """Retain all installed cases and adapt only two exact diagnostic checks."""
    definitions = namespace["TEST_DEFINITIONS"]
    by_name = {definition["name"]: definition for definition in definitions}
    if len(definitions) != 18 or len(by_name) != 18:
        raise ValueError(
            "Unexpected debug-agent suite registrations; refusing adaptation"
        )
    if (
        by_name["test_memory_violation"].get("patterns") != MEMORY_PATTERNS
        or by_name["test_sigquit_resumes"].get("patterns") != SIGQUIT_PATTERNS
    ):
        raise ValueError(
            "Unexpected debug-agent diagnostic assertions; refusing adaptation"
        )
    original = namespace["check_errors"]

    def check_errors(check_list, out_str, err_str):
        patterns = [check.pattern for check in check_list]
        if patterns in (MEMORY_PATTERNS, SIGQUIT_PATTERNS):
            memory_fault = patterns == MEMORY_PATTERNS
            # Keep the five fault checks, or the SIGQUIT kernel identity check.
            base_pass = original(check_list[:-1], out_str, err_str)
            pc_pass = valid_stopped_pc(err_str, memory_fault=memory_fault)
            if not pc_pass:
                namespace["logger"]().info(
                    "CSAN stopped-PC/dispatch/disassembly validation failed"
                )
            return base_pass and pc_pass
        return original(check_list, out_str, err_str)

    namespace["check_errors"] = check_errors
    original_loader = namespace["agent_loaded_by"]
    prepared_preload = os.environ.get("LD_PRELOAD", "")

    @contextlib.contextmanager
    def agent_loaded_by(mode):
        # Let the original context remove HSA_TOOLS_LIB and select the agent,
        # then retain the prepared interceptors and report initializer ahead of
        # that agent. The pinned agent starts from its static constructor and
        # has no hsa_init interposer (verified with the actual library on host).
        with original_loader(mode):
            if mode == "preload":
                if not prepared_preload:
                    raise ValueError("Missing prepared CSAN preload chain")
                agent_preload = os.environ.get("LD_PRELOAD", "")
                if not agent_preload:
                    raise ValueError("Installed harness did not select a preload agent")
                os.environ["LD_PRELOAD"] = prepared_preload + ":" + agent_preload
            yield

    namespace["agent_loaded_by"] = agent_loaded_by
    print(
        "CSAN debug-agent adaptation: preserve all 54 cases; validate memory-fault/SIGQUIT actual stopped PC",
        flush=True,
    )
    print(
        "CSAN debug-agent preload cases: retain CSAN/HSA/report chain and append the constructor-activated agent; GPU activation remains checked by the installed cases",
        flush=True,
    )


def adapted_harness(source: str) -> str:
    """Inject after registrations, before the installed top-level suite runs."""
    tree = ast.parse(source)
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "TEST_DEFINITIONS"
            for target in node.targets
        )
    ]
    checkers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "check_errors"
    ]
    if (
        len(definitions) != 1
        or len(checkers) != 1
        or [arg.arg for arg in checkers[0].args.args]
        != ["check_list", "out_str", "err_str"]
    ):
        raise ValueError("Unexpected installed debug-agent harness structure")
    definition = definitions[0]
    if not isinstance(definition.value, ast.List) or len(definition.value.elts) != 18:
        raise ValueError("Expected 18 installed debug-agent definitions")
    # Import this adapter by its actual repository path. The installed test
    # script is never imported (it runs tests at top level) or modified.
    injection = (
        "\nimport importlib.util as _csan_import\n"
        f"_csan_spec = _csan_import.spec_from_file_location('_therock_debug_adapter', {str(Path(__file__).resolve())!r})\n"
        "_csan_adapter = _csan_import.module_from_spec(_csan_spec)\n"
        "_csan_spec.loader.exec_module(_csan_adapter)\n"
        "_csan_adapter.adapt_installed_suite(globals())\n"
    )
    lines = source.splitlines(keepends=True)
    lines.insert(definition.end_lineno, injection)
    return "".join(lines)


def run_tests(artifacts_dir: Path, diagnostics_dir: Path) -> int:
    artifacts_dir = artifacts_dir.resolve()
    test_dir = artifacts_dir / "tests" / "rocm-debug-agent"
    script = test_dir / "run-test.py"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    if not script.is_file():
        print(f"Debug-agent test runner is missing: {script}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    for variable, directory in (
        ("PATH", artifacts_dir / "bin"),
        ("LD_LIBRARY_PATH", artifacts_dir / "lib"),
    ):
        env[variable] = str(directory) + (
            os.pathsep + env[variable] if env.get(variable) else ""
        )

    # Remove a previous diagnostic file so a failed launch cannot report stale
    # evidence. The installed harness opens this file in write mode itself.
    log_path = test_dir / "run-test.log"
    log_path.unlink(missing_ok=True)
    try:
        launch_script = script
        if env.get("CSAN_REPORT_PREFIX"):
            launch_script = diagnostics_dir.resolve() / "run-test-csan.py"
            launch_script.write_text(adapted_harness(script.read_text()))
        result = subprocess.run(
            [sys.executable, str(launch_script), str(test_dir)],
            cwd=test_dir,
            env=env,
            check=False,
        )
        exit_code = result.returncode
    except (OSError, ValueError, SyntaxError) as error:
        print(f"Cannot launch debug-agent tests: {error}", file=sys.stderr)
        exit_code = 1

    if log_path.is_file():
        destination = diagnostics_dir / "run-test.log"
        shutil.copyfile(log_path, destination)
        print(f"Debug-agent diagnostics preserved: {destination}", flush=True)
        if exit_code:
            print("--- debug-agent child diagnostics ---", flush=True)
            print(log_path.read_text(encoding="utf-8", errors="replace"), flush=True)
    else:
        print(
            f"Debug-agent runner produced no diagnostic log: {log_path}",
            file=sys.stderr,
        )

    # Convert a signal termination to the conventional shell exit status.
    return exit_code if exit_code >= 0 else 128 - exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_ARTIFACTS_DIR", "build")),
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=Path("test-diagnostics/rocr-debug-agent"),
    )
    args = parser.parse_args(argv)
    return run_tests(args.artifacts_dir, args.diagnostics_dir)


if __name__ == "__main__":
    sys.exit(main())
