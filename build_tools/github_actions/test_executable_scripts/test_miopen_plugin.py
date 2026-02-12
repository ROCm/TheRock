import logging
import os
import shlex
import subprocess
from pathlib import Path
import psutil



print("##[group]PSTREE_START")

mem = psutil.virtual_memory()
swap = psutil.swap_memory()
cpu_percent = psutil.cpu_percent(interval=1)  # interval=1 means sample over 1 second

print(f"CPU: {cpu_percent}%")
print(f"RAM:  {mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB ({mem.percent}%)")
print(f"Swap: {swap.used/1024**3:.1f}/{swap.total/1024**3:.1f} GB ({swap.percent}%)")

print("All process data")
for proc in psutil.process_iter(
    ['pid', 'name', 'ppid', 'memory_info', 'cpu_percent', 'create_time']
):
    print(proc.info)

print("##[endgroup]")


THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/miopen_legacy_plugin",
    "--verbose",
    "--parallel",
    "8",
    "--timeout",
    "1800",
]

# Determine test filter based on TEST_TYPE environment variable
environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "full")

if test_type == "smoke":
    # Exclude tests that start with "Full" during smoke tests
    environ_vars["GTEST_FILTER"] = "-Full*"

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
    env=environ_vars,
)


print("##[group]PSTREE_END")

mem = psutil.virtual_memory()
swap = psutil.swap_memory()
cpu_percent = psutil.cpu_percent(interval=1)  # interval=1 means sample over 1 second

print(f"CPU: {cpu_percent}%")
print(f"RAM:  {mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB ({mem.percent}%)")
print(f"Swap: {swap.used/1024**3:.1f}/{swap.total/1024**3:.1f} GB ({swap.percent}%)")

print("All process data")
for proc in psutil.process_iter(
    ['pid', 'name', 'ppid', 'memory_info', 'cpu_percent', 'create_time']
):
    print(proc.info)

print("##[endgroup]")
