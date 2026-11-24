# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bounded, supplementary GDB observations of executed CTest failures on Linux."""

import json
import os
import re
import selectors
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path


MAX_CASES = 8
MAX_CASE_SECONDS = 120
MAX_TOTAL_SECONDS = 600
MAX_LOG_BYTES = 2 * 1024 * 1024
MAX_INVENTORY_BYTES = 32 * 1024 * 1024
INVENTORY_SECONDS = 30

# Other properties may require CTest orchestration that a direct invocation loses.
SUPPORTED_PROPERTIES = {
    "COST",
    "DISABLED",
    "ENVIRONMENT",
    "ENVIRONMENT_MODIFICATION",
    "FAIL_REGULAR_EXPRESSION",
    "LABELS",
    "MEASUREMENT",
    "PASS_REGULAR_EXPRESSION",
    "PROCESSORS",
    "RUN_SERIAL",
    "SKIP_REGULAR_EXPRESSION",
    "SKIP_RETURN_CODE",
    "TIMEOUT",
    "WILL_FAIL",
    "WORKING_DIRECTORY",
}


def _kill_session(session: int) -> None:
    # GDB can put the inferior in a separate process group in the same session.
    # Killing only GDB's process group can leave a timed-out test running.
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rpartition(")")[2].split()
        except (FileNotFoundError, PermissionError):
            continue
        if int(fields[3]) == session:
            try:
                os.kill(int(stat.parent.name), signal.SIGKILL)
            except ProcessLookupError:
                pass


def _run_bounded(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    limit: int,
    gdb_commands: str = "",
) -> tuple[bytes, dict]:
    output = bytearray()
    status = {"status": "completed"}
    with tempfile.NamedTemporaryFile() as script:
        if gdb_commands:
            script.write(gdb_commands.encode())
            script.flush()
            arguments_index = command.index("--args")
            command = [
                *command[:arguments_index],
                "-x",
                script.name,
                *command[arguments_index:],
            ]
        with subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process:
            deadline = time.monotonic() + timeout
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while selector.get_map():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            status = {"status": "timeout", "timeout_seconds": timeout}
                            break
                        for key, _ in selector.select(min(remaining, 0.1)):
                            chunk = os.read(key.fd, min(65536, limit - len(output) + 1))
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            output.extend(chunk)
                            if len(output) > limit:
                                del output[limit:]
                                status = {
                                    "status": "output_limit",
                                    "limit_bytes": limit,
                                }
                                break
                        if status["status"] != "completed":
                            break
                if status["status"] == "completed":
                    try:
                        process.wait(timeout=max(0.01, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        status = {"status": "timeout", "timeout_seconds": timeout}
            finally:
                _kill_session(process.pid)
                process.wait()
            status["returncode"] = process.returncode
    return bytes(output), status


def _test_environment(base: dict[str, str], properties: dict) -> dict[str, str]:
    environment = {key: str(value) for key, value in base.items()}
    for entry in properties.get("ENVIRONMENT", []):
        key, separator, value = entry.partition("=")
        if not separator:
            raise ValueError(f"Invalid CTest ENVIRONMENT entry: {entry!r}")
        environment[key] = value
    unmodified = environment.copy()
    for entry in properties.get("ENVIRONMENT_MODIFICATION", []):
        key, separator, operation = entry.partition("=")
        operation, colon, value = operation.partition(":")
        if not separator or not colon:
            raise ValueError(f"Invalid CTest ENVIRONMENT_MODIFICATION entry: {entry!r}")
        if operation == "reset":
            if key in unmodified:
                environment[key] = unmodified[key]
            else:
                environment.pop(key, None)
        elif operation == "set":
            environment[key] = value
        elif operation == "unset":
            environment.pop(key, None)
        elif operation in ("string_append", "string_prepend"):
            previous = environment.get(key, "")
            environment[key] = (
                previous + value if operation.endswith("append") else value + previous
            )
        elif operation in (
            "path_list_append",
            "path_list_prepend",
            "cmake_list_append",
            "cmake_list_prepend",
        ):
            previous = environment.get(key, "")
            delimiter = ";" if operation.startswith("cmake") else os.pathsep
            environment[key] = (
                previous + delimiter + value
                if previous and operation.endswith("append")
                else value + delimiter + previous if previous else value
            )
        else:
            raise ValueError(f"Unsupported CTest environment operation: {operation}")
    return environment


def _gdb_commands(environment: dict[str, str], debugger_env: dict[str, str]) -> str:
    # GDB adds terminal dimensions to the inferior environment even in batch mode.
    commands = [
        f"unset environment {key}"
        for key in ("LINES", "COLUMNS")
        if key not in environment
    ]
    for key, value in sorted(environment.items()):
        if debugger_env.get(key) == value and key not in {"LINES", "COLUMNS"}:
            continue
        if (
            not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key)
            or any(character in value for character in "\r\n\0")
            or value != value.strip()
        ):
            raise ValueError(
                f"Environment entry cannot be represented literally in GDB: {key!r}"
            )
        commands.append(f"set environment {key} = {value}")
    commands.extend(
        [
            "run",
            "info program",
            "thread apply all bt full 64",
            "info sharedlibrary",
            "info proc mappings",
            "info registers",
            "x/i $pc",
        ]
    )
    return "\n".join(commands) + "\n"


def collect_inventory(
    command: list[str], *, cwd: Path, env: dict[str, str]
) -> tuple[bytes, dict]:
    # CTest --show-only truncates LastTest.log. Collect before the original run.
    return _run_bounded(
        command, cwd=cwd, env=env, timeout=INVENTORY_SECONDS, limit=MAX_INVENTORY_BYTES
    )


def capture_failures(
    *,
    junit: Path,
    inventory: bytes,
    inventory_status: dict,
    test_root: Path,
    env: dict[str, str],
) -> None:
    """Never alter CTest files or classify a debugger rerun as a test result.

    Capture the top-level command only. CTest JSON flattens launchers into argv;
    these observations do not qualify launcher or descendant-process behavior.
    """
    output = junit.parent / f"{junit.stem}-gdb"
    output.mkdir(exist_ok=False)
    report = {
        "scope": "Supplementary top-level process observations; original JUnit is authoritative",
        "limits": {
            "cases": MAX_CASES,
            "case_seconds": MAX_CASE_SECONDS,
            "total_seconds": MAX_TOTAL_SECONDS,
            "inventory_seconds": INVENTORY_SECONDS,
            "log_bytes": MAX_LOG_BYTES,
        },
        "cases": [],
    }
    deadline = time.monotonic() + MAX_TOTAL_SECONDS
    try:
        names = list(
            dict.fromkeys(
                case.attrib["name"]
                for case in ET.parse(junit).iter("testcase")
                if case.get("status") == "fail"
                and case.find("failure") is not None
                and case.find("skipped") is None
            )
        )
        if not names:
            report["selection"] = (
                "No failed executed cases in the original JUnit report"
            )
            return
        (output / "ctest-inventory.json").write_bytes(inventory)
        report["inventory"] = inventory_status
        if (
            inventory_status["status"] != "completed"
            or inventory_status["returncode"] != 0
        ):
            raise ValueError(
                "CTest JSON inventory unavailable; inspect ctest-inventory.json"
            )
        tests = json.loads(inventory)["tests"]
        root = test_root.resolve()
        for index, name in enumerate(names):
            entry = {"name": name}
            report["cases"].append(entry)
            try:
                if index >= MAX_CASES or time.monotonic() >= deadline:
                    raise ValueError("Diagnostic case or total-time budget exhausted")
                candidates = [test for test in tests if test["name"] == name]
                if len(candidates) != 1:
                    raise ValueError("Failed case has no unique CTest JSON command")
                test = candidates[0]
                properties = {
                    item["name"]: item["value"] for item in test.get("properties", [])
                }
                unsupported = properties.keys() - SUPPORTED_PROPERTIES
                if unsupported:
                    raise ValueError(
                        f"Unsupported CTest properties: {sorted(unsupported)}"
                    )
                if properties.get("DISABLED"):
                    raise ValueError("CTest JSON case is disabled")
                command = test["command"]
                directory = Path(properties.get("WORKING_DIRECTORY", root))
                if not directory.is_absolute():
                    raise ValueError("Relative CTest working directory is unsupported")
                executable = Path(command[0])
                if (
                    not executable.is_absolute()
                    or not executable.resolve().is_relative_to(root)
                ):
                    raise ValueError(
                        "Only direct executables inside the installed catch_tests tree are supported"
                    )
                with executable.open("rb") as binary:
                    if binary.read(4) != b"\x7fELF":
                        raise ValueError(
                            "Scripts and non-ELF test commands are unsupported"
                        )
                # This runner uses ordinary CTest mode, which sets this variable
                # before applying per-test ENVIRONMENT properties.
                environment = _test_environment(
                    env | {"CTEST_INTERACTIVE_DEBUG_MODE": "1"}, properties
                )
                entry.update(
                    command=command,
                    working_directory=str(directory),
                    properties=properties,
                )
                seconds = min(MAX_CASE_SECONDS, deadline - time.monotonic())
                if properties.get("TIMEOUT", 0) > 0:
                    seconds = min(seconds, properties["TIMEOUT"])
                if seconds <= 0:
                    raise ValueError("Diagnostic total-time budget exhausted")
                # The debugger itself must not preload the candidate's libraries.
                debugger_env = {
                    key: str(value)
                    for key, value in environment.items()
                    if not key.startswith("LD_")
                    and key not in {"PYTHONHOME", "PYTHONPATH"}
                }
                commands = _gdb_commands(environment, debugger_env)
                log, result = _run_bounded(
                    [
                        "/usr/bin/gdb",
                        "--nx",
                        "--nh",
                        "--quiet",
                        "--batch",
                        "-iex",
                        "set auto-load off",
                        "-iex",
                        "set debuginfod enabled off",
                        "-iex",
                        "set startup-with-shell off",
                        "-iex",
                        "set disable-randomization off",
                        "-iex",
                        "set pagination off",
                        "-iex",
                        "set confirm off",
                        "-ex",
                        "set print elements 32",
                        "-ex",
                        "set print max-depth 4",
                        "-ex",
                        "handle all nostop noprint pass",
                        "-ex",
                        "handle SIGSEGV stop print pass",
                        "--args",
                        *command,
                    ],
                    cwd=directory,
                    env=debugger_env,
                    timeout=seconds,
                    limit=MAX_LOG_BYTES,
                    gdb_commands=commands,
                )
                filename = f"case-{index + 1:03d}.txt"
                (output / filename).write_bytes(log)
                entry.update(result, log=filename)
                if result["status"] == "completed" and result["returncode"] != 0:
                    entry["status"] = "debugger_error"
            except (OSError, ValueError, KeyError, TypeError) as error:
                entry.update(status="unavailable", error=str(error))
    except (OSError, ValueError, KeyError, ET.ParseError) as error:
        report["error"] = str(error)
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
