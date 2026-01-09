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


def get_all_supported_devices(amdgpu_family: str = "") -> dict[str, list[int]]:
    """Detect supported AMDGPU devices and return mapping of arch to device indices.

    This function queries available GPUs and returns a mapping of architecture
    names to their system device indices. It does NOT set HIP_VISIBLE_DEVICES;
    callers should set it before running pytest.

    Args:
        amdgpu_family: AMDGPU family string. Can be:
            - Empty string (default): Auto-detect all visible GPUs supported by PyTorch
            - Specific arch (e.g., "gfx1151"): Find and use matching GPU
            - Wildcard family (e.g., "gfx94X"): Find all matching GPUs

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
        print(f"HIP_VISIBLE_DEVICES already set to: {current_hip_visible}")
    else:
        # HIP_VISIBLE_DEVICES not set, no remapping needed
        original_system_indices = None

    # Query both supported and visible GPUs in a single subprocess call
    # (doesn't initialize CUDA in main process)
    print("Getting GPU information from PyTorch...", end="")
    supported_gpus, raw_visible_gpus = get_supported_and_visible_gpus()
    print("done")

    # Normalize gpu names
    # get_supported_and_visible_gpus() (via device_properties.gcnArchName):
    # Often returns detailed arch names like "gfx942:sramecc+:xnack-" or "gfx1100:xnack-"
    visible_gpus = [gpu.split(":")[0] for gpu in raw_visible_gpus]

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

        print(
            f"AMDGPU Arch detected via wildcard match '{partial_match}': "
            f"{selected_gpu_archs} (logical indices {selected_gpu_indices})"
        )
    else:
        # Mode 3: Specific GPU arch - validate it is visible and supported by the current PyTorch build.

        # We have gfx1151 -> we want to match exactly gfx1151
        # We have gfx950-dcgpu -> we need to match exactly gfx950
        # So remove the suffix after '-'
        pruned_amdgpu_family = amdgpu_family.split("-")[0]
        for idx, gpu in enumerate(visible_gpus):
            if gpu in supported_gpus:
                if gpu == pruned_amdgpu_family or pruned_amdgpu_family in gpu:
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

    print(f"Detected PyTorch supported architecture at device indices: {result}")
    return result


def get_unique_supported_devices(amdgpu_family: str = "") -> dict[str, list[int]]:
    """
    Returns a dictionary mapping each supported architecture to a single device index (the first one for each).
    This is a convenience wrapper over get_all_supported_devices for situations where
    only one device per arch is desired.

    Args:
        amdgpu_family: Optionally filter by a specific AMDGPU family string or pattern.

    Returns:
        Dictionary: {arch: [device_index]} for each supported arch (single-element lists).
    """
    devices_by_arch = get_all_supported_devices(amdgpu_family)
    unique_devices = {
        arch: [indices[0]] for arch, indices in devices_by_arch.items() if indices
    }
    return unique_devices


def set_gpu_execution_policy(
    supported_devices: dict[str, list[int]],
    policy: str,
    offset: int = 0,
) -> list[tuple[str, int]]:
    """
    Configures the HIP_VISIBLE_DEVICES environment variable according to a GPU selection policy,
    enabling targeted execution on specific AMD GPUs for PyTorch/pytest runs. This must be run
    *before* torch is imported, because HIP_VISIBLE_DEVICES cannot affect CUDA device visibility after initialization.

    Args:
        supported_devices (dict[str, list[int]]): Dictionary mapping GPU architectures to lists of device indices.
            Can be obtained from get_all_supported_devices() or get_unique_supported_devices().
            - get_all_supported_devices(): {"gfx942": [0, 1], "gfx1100": [2]}
            - get_unique_supported_devices(): {"gfx942": [0], "gfx1100": [2]}
        policy (str): Device selection policy. Must be one of:
            - "single": Use a single device from the provided devices at the given offset.
            - "all": Use all provided devices. The offset parameter is ignored.
        offset (int): Index offset for selecting device with "single" policy. Ignored for "all" policy.
            Offset is applied to the flattened list of (arch, idx) pairs made from supported_devices dictionary.
            Depending on the function used to get supported_devices:
            - get_all_supported_devices(): Select the device at the given offset.
              Example: {"gfx942": [0, 1], "gfx1100": [2]} => [("gfx942", 0), ("gfx942", 1), ("gfx1100", 2)]
            - get_unique_supported_devices(): Since every architecture has a single device,
              offset effectively allows us to iterate over specific architectures.
              Example: {"gfx942": [0], "gfx1100": [2]} => [("gfx942", 0), ("gfx1100", 2)]

    Returns:
        list[tuple[str, int]]: A list of (arch, device_index) tuples that were selected and made visible.
            - For policy "single", the list contains a single (arch, idx).
            - For "all", the list contains every (arch, idx) made visible.

    Raises:
        ValueError: If an invalid policy is supplied or if supported_devices is empty.
        IndexError: If the requested offset exceeds the set of possible devices (only for "single" policy).
    """
    valid_policies = ("single", "all")
    if policy not in valid_policies:
        raise ValueError(f"Invalid policy '{policy}'. Must be one of {valid_policies}.")

    if not supported_devices:
        raise ValueError("supported_devices cannot be empty; no devices available.")

    if policy == "single":
        # Flatten the supported_devices dictionary into pairs of (arch, idx) for each device.
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
        print(f"Policy '{policy}': Using device {device_idx} ({arch})")
        return [(arch, device_idx)]

    else:
        # "all" policy: Use all supported devices in the provided dictionary
        # Can have multiple devices per specific architecture
        flat_devices = [
            (arch, idx)
            for arch, indices in supported_devices.items()
            for idx in indices
        ]
        device_indices_str = ",".join(str(idx) for _, idx in flat_devices)
        os.environ["HIP_VISIBLE_DEVICES"] = device_indices_str
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
