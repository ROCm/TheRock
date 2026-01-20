"""Shared utilities for PyTorch testing."""

import os
import subprocess
import sys

from importlib.metadata import version


def get_supported_and_visible_gpus() -> tuple[list[str], list[str]]:
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


def get_all_supported_devices(
    amdgpu_family: str = "", log: bool = True
) -> dict[str, list[int]]:
    """Detect supported AMDGPU devices and return mapping of arch to device indices.

    This function queries available GPUs and returns a mapping of architecture
    names to their system device indices. It does NOT set HIP_VISIBLE_DEVICES;
    callers should set it before running pytest.

    Args:
        amdgpu_family: AMDGPU family string. Can be:
            - Empty string (default): Auto-detect all visible GPUs supported by PyTorch
            - Specific arch (e.g., "gfx1151"): Find and use matching GPU
            - Wildcard family (e.g., "gfx94X"): Find all matching GPUs
        log: If True, prints summary and debug information to the console.

    Returns:
        Dictionary mapping architecture names to lists of system device indices.
        E.g., {"gfx942": [0, 1], "gfx1100": [2]}
        Exits on failure.

    Side effects:
        - Reads HIP_VISIBLE_DEVICES if already set (respects pre-configured constraints)
        - If used to set HIP_VISIBLE_DEVICES, this must be done before importing torch in the main process via pytest
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
        if log:
            print(f"HIP_VISIBLE_DEVICES already set to: {current_hip_visible}")
    else:
        # HIP_VISIBLE_DEVICES not set, no remapping needed
        original_system_indices = None

    # Query both supported and visible GPUs in a single subprocess call
    # (doesn't initialize CUDA in main process)
    if log:
        print("Getting GPU information from PyTorch...", end="")
    supported_gpus, raw_visible_gpus = get_supported_and_visible_gpus()
    if log:
        print("done")

    # Normalize gpu names
    # get_supported_and_visible_gpus() (via device_properties.gcnArchName):
    # Often returns detailed arch names like "gfx942:sramecc+:xnack-" or "gfx1100:xnack-"
    visible_gpus = [gpu.split(":")[0] for gpu in raw_visible_gpus]

    if log:
        print(f"Supported AMD GPUs: {supported_gpus}")
        print(f"Visible AMD GPUs: {visible_gpus}")

    selected_gpu_indices = []
    selected_gpu_archs = []

    if not amdgpu_family:
        # Mode 1: Auto-detect - use all supported GPUs
        for idx, gpu in enumerate(visible_gpus):
            if gpu in supported_gpus:
                selected_gpu_indices.append(idx)
                selected_gpu_archs.append(gpu)
        if len(selected_gpu_archs) == 0:
            print("[ERROR] No GPU found in visible GPUs that is supported by PyTorch")
            sys.exit(1)
    elif amdgpu_family.split("-")[0].upper().endswith("X"):
        # Mode 2: Wildcard match (e.g., "gfx94X" matches "gfx942", "gfx940", etc.)
        family_part = amdgpu_family.split("-")[0]
        partial_match = family_part[:-1]  # Remove the 'X'

        for idx, gpu in enumerate(visible_gpus):
            if partial_match in gpu and gpu in supported_gpus:
                selected_gpu_indices.append(idx)
                selected_gpu_archs.append(gpu)

        if len(selected_gpu_archs) == 0:
            print(f"[ERROR] No GPU found matching wildcard pattern '{amdgpu_family}'.")
            sys.exit(1)

        if log:
            print(
                f"AMDGPU Arch detected via wildcard match '{partial_match}': "
                f"{selected_gpu_archs} (logical indices {selected_gpu_indices})"
            )
    else:
        # Mode 3: Specific GPU arch - validate it is visible and supported by the current PyTorch build.
        for idx, gpu in enumerate(visible_gpus):
            if gpu in supported_gpus:
                if gpu == amdgpu_family or amdgpu_family in gpu:
                    selected_gpu_indices.append(idx)
                    selected_gpu_archs.append(gpu)

        if len(selected_gpu_archs) == 0:
            print(
                f"[ERROR] Requested GPU '{amdgpu_family}' not found in visible GPUs that are supported by PyTorch"
            )
            sys.exit(1)

    # Map logical indices back to system indices if HIP_VISIBLE_DEVICES was already set
    if original_system_indices is not None:
        # Map: logical index -> original system index
        # e.g., if HIP_VISIBLE_DEVICES="2,3,4" and we selected logical index 0,
        # the system index is 2 (the original system index)
        system_gpu_indices = [
            original_system_indices[idx] for idx in selected_gpu_indices
        ]
    else:
        # HIP_VISIBLE_DEVICES not set, no remapping needed
        system_gpu_indices = selected_gpu_indices

    # Build the result dictionary: arch -> list of system device indices
    result = {}
    for arch, sys_idx in zip(selected_gpu_archs, system_gpu_indices):
        if arch not in result:
            result[arch] = []
        result[arch].append(sys_idx)

    if log:
        print(f"Detected PyTorch supported architecture at device indices: {result}")
    return result


def get_unique_supported_devices(
    amdgpu_family: str = "", log: bool = False
) -> dict[str, int]:
    """
    Returns a dictionary mapping each supported architecture to a single device index (the first one for each).
    This is a convenience wrapper over get_all_supported_devices for situations where
    only one device per arch is desired.

    Args:
        amdgpu_family: Optionally filter by a specific AMDGPU family string or pattern.
        log: If True, passes through to get_all_supported_devices for printing.

    Returns:
        Dictionary: {arch: device_index} for each supported arch.
    """
    devices_by_arch = get_all_supported_devices(amdgpu_family, log=log)
    unique_devices = {
        arch: indices[0] for arch, indices in devices_by_arch.items() if indices
    }
    return unique_devices


def get_unique_supported_devices_count(
    amdgpu_family: str = "", log: bool = False
) -> int:
    """Get the number of unique supported architectures.

    Args:
        amdgpu_family: AMDGPU family filter string (optional).
        log: If True, passes through to get_unique_supported_devices for printing.

    Returns:
        Count of unique architectures (one device per arch).
    """
    unique_devices_per_arch = get_unique_supported_devices(amdgpu_family, log=log)
    return len(unique_devices_per_arch)


def set_gpu_execution_policy(
    amdgpu_family: str, policy: str, offset: int = 0, log: bool = True
) -> list[tuple[str, int]]:
    """
    Configures the HIP_VISIBLE_DEVICES environment variable according to a GPU selection policy,
    enabling targeted execution on specific AMD GPUs for PyTorch/pytest runs. This must be run
    *before* torch is imported, because HIP_VISIBLE_DEVICES cannot affect CUDA device visibility after initialization.

    Args:
        amdgpu_family (str): (Optional) AMDGPU family filter string, e.g., "gfx942", "gfx94X", or "" for all.
        policy (str): Device selection policy. Must be one of:
            - "single": Use a single device from all supported devices at the given offset.
            - "unique-single": Use a single device from the set of unique architectures at the given offset.
            - "unique": Use the first device for each detected unique architecture (all at once).
            - "all": Use all supported devices (every detected, possibly multiple per arch).
        offset (int): Index offset for selecting device in "single" or "unique-single" mode.
        log (bool): If True, prints device selection details.

    Returns:
        list[tuple[str, int]]: A list of (arch, device_index) tuples that were selected and made visible.
            - For policies "single" and "unique-single", the list contains a single (arch, idx).
            - For "unique" and "all", the list contains every (arch, idx) made visible.

    Raises:
        ValueError: If an invalid policy is supplied.
        IndexError: If the requested offset exceeds the set of possible devices.
    """
    valid_policies = ("single", "unique-single", "unique", "all")
    if policy not in valid_policies:
        raise ValueError(f"Invalid policy '{policy}'. Must be one of {valid_policies}.")

    if policy in ("unique", "unique-single"):
        supported_devices = get_unique_supported_devices(amdgpu_family, log=log)
    else:
        supported_devices = get_all_supported_devices(amdgpu_family, log=log)

    if not supported_devices:
        print("[ERROR] No supported devices found")
        sys.exit(1)

    if policy == "single":
        # Flatten all (arch, idx) pairs and select using offset.
        flat_devices = [
            (arch, idx)
            for arch, indices in supported_devices.items()
            for idx in indices
        ]
        if offset < 0 or offset >= len(flat_devices):
            raise IndexError(
                f"Offset {offset} out of range for {len(flat_devices)} total devices"
            )
        arch, device_idx = flat_devices[offset]
        os.environ["HIP_VISIBLE_DEVICES"] = str(device_idx)
        if log:
            print(f"Policy '{policy}': Using device {device_idx} ({arch})")
        return [(arch, device_idx)]

    elif policy == "unique-single":
        # Selects a single device (first device) from unique architectures using offset.
        flat_unique_devices = [(arch, idx) for arch, idx in supported_devices.items()]
        if offset < 0 or offset >= len(flat_unique_devices):
            raise IndexError(
                f"Offset {offset} out of range for {len(flat_unique_devices)} unique devices"
            )
        arch, device_idx = flat_unique_devices[offset]
        os.environ["HIP_VISIBLE_DEVICES"] = str(device_idx)
        if log:
            print(f"Policy '{policy}': Using device {device_idx} ({arch})")
        return [(arch, device_idx)]

    elif policy == "unique":
        # Use one device per architecture (first device of each arch) simultaneously
        flat_devices = [(arch, idx) for arch, idx in supported_devices.items()]
        device_indices_str = ",".join(str(idx) for _, idx in flat_devices)
        os.environ["HIP_VISIBLE_DEVICES"] = device_indices_str
        if log:
            device_pairs_str = ", ".join(f"{arch}: {idx}" for arch, idx in flat_devices)
            print(f"Policy '{policy}': Using devices [{device_pairs_str}]")
        return flat_devices

    else:
        # "all" policy: Use all supported devices (can have multiple per arch)
        flat_devices = [
            (arch, idx)
            for arch, indices in supported_devices.items()
            for idx in indices
        ]
        device_indices_str = ",".join(str(idx) for _, idx in flat_devices)
        os.environ["HIP_VISIBLE_DEVICES"] = device_indices_str
        if log:
            device_pairs_str = ", ".join(f"{arch}: {idx}" for arch, idx in flat_devices)
            print(f"Policy 'all': Using devices [{device_pairs_str}]")
        return flat_devices


def detect_pytorch_version() -> str:
    """Auto-detect the PyTorch version from the installed package.

    Returns:
        The detected PyTorch version as major.minor (e.g., "2.7").
    """
    # Get version, remove build suffix (+rocm, +cpu, etc.) and patch version
    return version("torch").rsplit("+", 1)[0].rsplit(".", 1)[0]
