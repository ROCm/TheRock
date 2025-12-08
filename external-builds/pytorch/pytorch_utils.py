"""Shared utilities for PyTorch testing."""

import os
import subprocess
import sys

from importlib.metadata import version


def get_visible_gpus() -> list[str]:
    """Get a list of GPUs that are visible for torch.

    Note that the current torch build does not necessarily have
    support for all of the GPUs that are visible.
    The list of GPUs that are supported by the current torch build
    can be queried with method torch.cuda.get_arch_list().

    This function runs in a subprocess to avoid initializing CUDA
    in the main process before HIP_VISIBLE_DEVICES is set.

    Important: If HIP_VISIBLE_DEVICES is already set before calling this script,
    this function will only see GPUs within that constraint. This allows the
    script to work within pre-configured limitations (e.g., in containers).

    Returns:
        List of AMDGPU family strings visible (e.g., ["gfx942", "gfx1100"]).
        Exits on failure.
    """
    query_script = """
import sys
try:
    import torch
    visible_gpus = []
    if not torch.cuda.is_available():
        print("ERROR:ROCm is not available", file=sys.stderr)
        sys.exit(1)

    gpu_count = torch.cuda.device_count()
    print(f"GPU count visible for PyTorch: {gpu_count}", file=sys.stderr)

    for device_idx in range(gpu_count):
        device_id = f"cuda:{device_idx}"
        device = torch.cuda.device(device_id)
        if device:
            device_properties = torch.cuda.get_device_properties(device)
            if device_properties and hasattr(device_properties, 'gcnArchName'):
                # AMD GPUs have gcnArchName
                visible_gpus.append(device_properties.gcnArchName)

    if len(visible_gpus) == 0:
        print("No AMD GPUs with gcnArchName detected", file=sys.stderr)
        sys.exit(1)

    # Print one GPU per line for easy parsing
    for gpu in visible_gpus:
        print(gpu)
except Exception as e:
    print(f"{e}", file=sys.stderr)
    sys.exit(1)
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", query_script],
            capture_output=True,
            text=True,
            check=True,
        )
        visible_gpus = result.stdout.strip().split("\n")
        return visible_gpus
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to retrieve visible GPUs: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error retrieving visible GPUs: {e}")
        sys.exit(1)


def get_supported_gpus() -> list[str]:
    """Get a list of AMD GPUs that are supported by the current PyTorch build.

    Returns:
        List of PyTorch supported GPU architecture strings (e.g., ["gfx942", "gfx1100"]).
        Exits on failure.
    """
    query_script = """
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print("ROCm is not available", file=sys.stderr)
        sys.exit(1)
    gpus = torch.cuda.get_arch_list()
    if len(gpus) == 0:
        print("No AMD GPUs detected", file=sys.stderr)
        sys.exit(1)
    # Print one GPU per line for easy parsing
    for gpu in gpus:
        print(gpu)
except Exception as e:
    print(f"ERROR:{e}", file=sys.stderr)
    sys.exit(1)
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", query_script],
            capture_output=True,
            text=True,
            check=True,
        )
        available_gpus = result.stdout.strip().split("\n")
        return available_gpus
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to retrieve available GPUs: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error retrieving available GPUs: {e}")
        sys.exit(1)


def get_gpu_info() -> tuple[list[str], list[str]]:
    """Get both supported and visible GPUs in a single subprocess call.

    Note that the current torch build does not necessarily have
    support for all of the GPUs that are visible.

    This function runs in a subprocess to avoid initializing CUDA
    in the main process before HIP_VISIBLE_DEVICES is set.

    Important: If HIP_VISIBLE_DEVICES is already set before calling this script,
    this function will only see GPUs within that constraint. This allows the
    script to work within pre-configured limitations (e.g., in containers).

    Returns:
        Tuple of (supported_gpus, visible_gpus):
            - supported_gpus: List of AMDGPU archs supported by PyTorch build
            - visible_gpus: List of AMDGPU archs physically visible
        Exits on failure.
    """
    query_script = """
import sys
try:
    import torch

    if not torch.cuda.is_available():
        print("ERROR:ROCm is not available", file=sys.stderr)
        sys.exit(1)

    # Get supported AMDGPUs (from PyTorch build)
    supported_gpus = torch.cuda.get_arch_list()
    if len(supported_gpus) == 0:
        print("ERROR:No AMD GPUs in PyTorch build", file=sys.stderr)
        sys.exit(1)

    # Get visible GPUs (from hardware)
    visible_gpus = []
    gpu_count = torch.cuda.device_count()
    print(f"GPU count visible for PyTorch: {gpu_count}", file=sys.stderr)

    for device_idx in range(gpu_count):
        device_id = f"cuda:{device_idx}"
        device = torch.cuda.device(device_id)
        if device:
            device_properties = torch.cuda.get_device_properties(device)
            if device_properties and hasattr(device_properties, 'gcnArchName'):
                # AMD GPUs have gcnArchName
                visible_gpus.append(device_properties.gcnArchName)

    if len(visible_gpus) == 0:
        print("ERROR:No AMD GPUs with gcnArchName detected", file=sys.stderr)
        sys.exit(1)

    # Output format: SUPPORTED|gpu1,gpu2,gpu3
    #                VISIBLE|gpu1,gpu2,gpu3
    print(f"SUPPORTED|{','.join(supported_gpus)}")
    print(f"VISIBLE|{','.join(visible_gpus)}")

except Exception as e:
    print(f"ERROR:{e}", file=sys.stderr)
    sys.exit(1)
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", query_script],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse the output
        lines = result.stdout.strip().split("\n")
        supported_gpus = []
        visible_gpus = []

        for line in lines:
            if line.startswith("SUPPORTED|"):
                supported_gpus = line.split("|")[1].split(",")
            elif line.startswith("VISIBLE|"):
                visible_gpus = line.split("|")[1].split(",")

        if not supported_gpus or not visible_gpus:
            print(f"\n[ERROR] Failed to parse GPU info from subprocess")
            sys.exit(1)

        return supported_gpus, visible_gpus

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to retrieve GPU info: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error retrieving GPU info: {e}")
        sys.exit(1)


def detect_amdgpu_family(amdgpu_family: str = "") -> list[str]:
    """Detect and configure AMDGPU family for testing.

    This function queries available GPUs and sets HIP_VISIBLE_DEVICES BEFORE
    PyTorch/CUDA is initialized in the main process via pytest.

    Args:
        amdgpu_family: AMDGPU family string. Can be:
            - Empty string (default): Auto-detect first visible GPU supported by PyTorch
            - Specific arch (e.g., "gfx1151"): Find and use matching GPU
            - Wildcard family (e.g., "gfx94X"): Find all matching GPUs

    Returns:
        List of detected AMDGPU family strings. Exits on failure.

    Side effects:
        - Reads HIP_VISIBLE_DEVICES if already set (respects pre-configured constraints)
        - Updates HIP_VISIBLE_DEVICES to further filter GPU selection
        - This MUST be called before importing torch in the main process via pytest
    """

    # Get the current HIP_VISIBLE_DEVICES to properly map indices
    # If already set (e.g., "2,3,4"), visible GPU indices are remapped (0,1,2)
    # We need to track the original system indices for correct remapping
    current_hip_visible = os.environ.get("HIP_VISIBLE_DEVICES", "")
    if current_hip_visible:
        # Parse existing HIP_VISIBLE_DEVICES to get original system GPU indices
        original_system_indices = [
            int(idx.strip()) for idx in current_hip_visible.split(",")
        ]
        print(f"HIP_VISIBLE_DEVICES already set to: {current_hip_visible}")
    else:
        # HIP_VISIBLE_DEVICES not set, no remapping needed
        original_system_indices = None

    # Query both supported and visible GPUs in a single subprocess call
    # (doesn't initialize CUDA in main process)
    print("Getting GPU information from PyTorch...", end="")
    supported_gpus, raw_visible_gpus = get_gpu_info()
    print("done")

    # Normalize gpu names
    # get_gpu_info() (via device_properties.gcnArchName):
    # Often returns detailed arch names like "gfx942:sramecc+:xnack-" or "gfx1100:xnack-"
    visible_gpus = [gpu.split(":")[0] for gpu in raw_visible_gpus]

    print(f"Supported AMD GPUs: {supported_gpus}")
    print(f"Visible AMD GPUs: {visible_gpus}")

    selected_gpu_indices = []
    selected_gpu_archs = []

    if not amdgpu_family:
        # Mode 1: Auto-detect - use first supported GPU
        for idx, gpu in enumerate(visible_gpus):
            if gpu in supported_gpus:
                selected_gpu_indices = [idx]
                selected_gpu_archs = [gpu]
                break
        if len(selected_gpu_archs) == 0:
            print(f"[ERROR] No GPU found in visible GPUs that is supported by PyTorch")
            sys.exit(1)
        print(
            f"AMDGPU Arch auto-detected (using GPU at logical index {selected_gpu_indices[0]}): {selected_gpu_archs[0]}"
        )
    elif amdgpu_family.split("-")[0].upper().endswith("X"):
        # Mode 2: Wildcard match (e.g., "gfx94X" matches "gfx942", "gfx940", etc.)
        family_part = amdgpu_family.split("-")[0]
        partial_match = family_part[:-1]  # Remove the 'X'

        for idx, gpu in enumerate(visible_gpus):
            if partial_match in gpu and gpu in supported_gpus:
                selected_gpu_indices += [idx]
                selected_gpu_archs += [gpu]

        if len(selected_gpu_archs) == 0:
            print(f"[ERROR] No GPU found matching wildcard pattern '{amdgpu_family}'.")
            sys.exit(1)

        print(
            f"AMDGPU Arch detected via wildcard match '{partial_match}': "
            f"{selected_gpu_archs} (logical indices {selected_gpu_indices})"
        )
    else:
        # Mode 3: Specific GPU arch - validate it is visible and supported by the current PyTorch build.
        for idx, gpu in enumerate(visible_gpus):
            if gpu in supported_gpus:
                if gpu == amdgpu_family or amdgpu_family in gpu:
                    selected_gpu_indices += [idx]
                    selected_gpu_archs += [gpu]

        if len(selected_gpu_archs) == 0:
            print(
                f"[ERROR] Requested GPU '{amdgpu_family}' not found in visible GPUs that are supported by PyTorch"
            )
            sys.exit(1)

        print(
            f"AMDGPU Arch validated: {selected_gpu_archs} (logical indices {selected_gpu_indices})"
        )

    # Set HIP_VISIBLE_DEVICES to select the specific GPU(s)
    # This MUST be done before torch is imported in the main process via pytest.

    # Map logical indices back to system indices if HIP_VISIBLE_DEVICES was already set
    if original_system_indices is not None:
        # Map: logical index -> original system index
        # e.g., if HIP_VISIBLE_DEVICES="2,3,4" and we selected logical index 0,
        # we need to set HIP_VISIBLE_DEVICES="2" (the original system index)
        system_gpu_indices = [
            original_system_indices[idx] for idx in selected_gpu_indices
        ]
    else:
        # HIP_VISIBLE_DEVICES not set, no remapping needed
        system_gpu_indices = selected_gpu_indices

    str_indices = ",".join(str(idx) for idx in system_gpu_indices)
    os.environ["HIP_VISIBLE_DEVICES"] = str_indices
    print(f"Set HIP_VISIBLE_DEVICES={str_indices}")

    return selected_gpu_archs


def detect_pytorch_version() -> str:
    """Auto-detect the PyTorch version from the installed package.

    Returns:
        The detected PyTorch version as major.minor (e.g., "2.7").
    """
    # Get version, remove build suffix (+rocm, +cpu, etc.) and patch version
    return version("torch").rsplit("+", 1)[0].rsplit(".", 1)[0]
