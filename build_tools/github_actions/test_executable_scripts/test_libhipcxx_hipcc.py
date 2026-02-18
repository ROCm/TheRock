import json
import logging
import os
import shlex
import sys
import subprocess
from pathlib import Path
import platform

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# get GPU architecture
def get_current_gpu_architecture():
    """
    Execute offload-arch command and return the second line of output.

    Returns:
        str: The second line of the offload-arch output, or None if not available.
    """
    try:
        subprocess.run(
            f"python {THEROCK_DIR}/build_tools/setup_venv.py --index-name nightly --index-subdir gfx110X-all --packages rocm .tmpvenv",
            shell=True,
        )
        if platform.system() == "Windows":
            offload_arch_location = ".tmpvenv/Scripts/offload-arch.exe"
        else:
            offload_arch_location = ".tmpvenv/bin/offload-arch"
        result = subprocess.run(
            [offload_arch_location], capture_output=True, text=True, check=True
        )

        lines = result.stdout.strip().split("\n")
        logging.info(f"DEBUG:{lines}")

        if len(lines) >= 2:
            return lines[1]
        else:
            print(f"Warning: offload-arch returned fewer than 2 lines", file=sys.stderr)
            return lines[-1]

    except subprocess.CalledProcessError as e:
        print(f"Error executing offload-arch: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("Error: offload-arch command not found", file=sys.stderr)
        return None


gpu_arch = get_current_gpu_architecture()
logging.info(f"++ Detected GPU architecture: {gpu_arch}")


# Load ROCm version from version.json
def load_rocm_version() -> str:
    """Loads the rocm-version from the repository's version.json file."""
    version_file = THEROCK_DIR / "version.json"
    logging.info(f"Loading ROCm version from: {version_file}")
    with open(version_file, "rt") as f:
        loaded_file = json.load(f)
        return loaded_file["rocm-version"]


ROCM_VERSION = load_rocm_version()
logging.info(f"ROCm version: {ROCM_VERSION}")

environ_vars = os.environ.copy()


def prepend_env_path(env: dict, var_name: str, new_path: str):
    existing = env.get(var_name)
    if existing:
        env[var_name] = f"{new_path}{os.pathsep}{existing}"
    else:
        env[var_name] = new_path


# Resolve absolute paths
OUTPUT_ARTIFACTS_PATH = Path(OUTPUT_ARTIFACTS_DIR).resolve()
THEROCK_BIN_PATH = Path(THEROCK_BIN_DIR).resolve()

# Set up ROCm/HIP environment
environ_vars["ROCM_PATH"] = str(OUTPUT_ARTIFACTS_PATH)
environ_vars["HIP_DEVICE_LIB_PATH"] = str(
    OUTPUT_ARTIFACTS_PATH / "lib/llvm/amdgcn/bitcode/"
)
environ_vars["HIP_PATH"] = str(OUTPUT_ARTIFACTS_PATH)
environ_vars["CMAKE_PREFIX_PATH"] = str(OUTPUT_ARTIFACTS_PATH)
environ_vars["HIP_PLATFORM"] = "amd"
environ_vars["ROCM_VERSION"] = str(ROCM_VERSION)
environ_vars["CMAKE_GENERATOR"] = "Ninja"

# Add ROCm binaries to PATH
rocm_bin = str(THEROCK_BIN_PATH)
prepend_env_path(environ_vars, "PATH", rocm_bin)

# Set library paths
rocm_lib = str(OUTPUT_ARTIFACTS_PATH / "lib")
prepend_env_path(environ_vars, "LD_LIBRARY_PATH", rocm_lib)

logging.info(f"ROCM_PATH: {environ_vars['ROCM_PATH']}")
logging.info(f"HIP_PATH: {environ_vars['HIP_PATH']}")
logging.info(f"PATH: {environ_vars['PATH']}")

LIBHIPCXX_BUILD_DIR = OUTPUT_ARTIFACTS_PATH / "libhipcxx"

try:
    os.chdir(LIBHIPCXX_BUILD_DIR)
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    os.chdir(build_dir)
    logging.info(f"Changed working directory to: {os.getcwd()}")
except FileNotFoundError as e:
    logging.error(f"Error: Directory '{LIBHIPCXX_BUILD_DIR}' does not exist.")
    raise

if platform.system() == "Windows":
    HIPCC_BINARY_NAME = "hipcc.exe"
elif platform.system() == "Linux":
    HIPCC_BINARY_NAME = "hipcc"
else:
    print("Incompatible platform!")

HIP_COMPILER_ROCM_ROOT = OUTPUT_ARTIFACTS_PATH
if platform.system() == "Windows":
    environ_vars["HIPCXX"] = str(
        OUTPUT_ARTIFACTS_PATH / "lib" / "llvm" / "bin" / "amdclang++.exe"
    )
    print("HIPCXX:", environ_vars["HIPCXX"], str(OUTPUT_ARTIFACTS_PATH))

# Configure with CMake
cmd = [
    "cmake",
    f"-DCMAKE_PREFIX_PATH={OUTPUT_ARTIFACTS_PATH}",
    f"-DHIP_HIPCC_EXECUTABLE={(THEROCK_BIN_PATH / HIPCC_BINARY_NAME).as_posix()}",
    f"-DCMAKE_CXX_COMPILER={(THEROCK_BIN_PATH / HIPCC_BINARY_NAME)}",
    f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={HIP_COMPILER_ROCM_ROOT.as_posix()}",
    f"-DCMAKE_HIP_ARCHITECTURES={gpu_arch}",
]

# Find lit executable (from venv or system)
if platform.system() == "Windows":
    lit_executable = None
    # Check in current venv
    venv_lit = Path(os.sys.prefix) / "Scripts" / "lit.exe"
    if venv_lit.exists():
        lit_executable = str(venv_lit)
    # Check in .tmpvenv
    elif (Path(".tmpvenv") / "Scripts" / "lit.exe").exists():
        lit_executable = str((Path(".tmpvenv") / "Scripts" / "lit.exe").resolve())
    if lit_executable:
        logging.info(f"Found lit executable at: {lit_executable}")
        cmd.append(f"-Dlibcudacxx_LIT={lit_executable}")
        cmd.append(f"-DLLVM_EXTERNAL_LIT={lit_executable}")

# Add rc compiler for windows
if platform.system() == "Windows":
    cmd.append("-DCMAKE_RC_COMPILER=rc.exe")

cmd.extend(["-GNinja", ".."])

logging.info(f"++ Exec [{os.getcwd()}]$ {shlex.join(cmd)}")
subprocess.run(cmd, check=True, env=environ_vars)
# Run the tests using lit
if platform.system() == "Windows":
    cmd = [
        "ninja",
        "check-hipcxx",
    ]
else:
    cmd = [
        "bash",
        "../ci/test_libhipcxx.sh",
        "-cmake-options",
        f"-DHIP_HIPCC_EXECUTABLE={THEROCK_BIN_PATH / HIPCC_BINARY_NAME} -DCMAKE_HIP_COMPILER_ROCM_ROOT={HIP_COMPILER_ROCM_ROOT} -DCMAKE_HIP_ARCHITECTURES={gpu_arch}",
    ]
logging.info(f"++ Exec [{os.getcwd()}]$ {shlex.join(cmd)}")

subprocess.run(cmd, check=True, env=environ_vars)
