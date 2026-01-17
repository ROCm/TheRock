"""Hardware detection for CPU and GPU using system commands and ROCm tools."""

import subprocess
import re
import os
import json
import platform
import multiprocessing
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

from ..logger import log


@dataclass
class CpuInfo:
    """CPU information dataclass with model, cores, cache, and clock details."""

    model_name: str = "Unknown"
    cores: int = 0
    sockets: int = 1
    ram_size_gb: int = 0
    numa_nodes: int = 1
    clock_speed_mhz: int = 0
    l1_cache_kb: int = 0
    l2_cache_kb: int = 0
    l3_cache_kb: int = 0

    def getCpuModelName(self) -> str:
        """Get CPU model name."""
        return self.model_name

    def getCpuCores(self) -> int:
        """Get CPU cores count."""
        return self.cores

    def getCpuSockets(self) -> int:
        """Get CPU sockets count."""
        return self.sockets

    def getCpuRamSize(self) -> int:
        """Get RAM size in GB."""
        return self.ram_size_gb

    def getCpuNumaNodes(self) -> int:
        """Get NUMA nodes count."""
        return self.numa_nodes

    def getCpuClockSpeed(self) -> int:
        """Get CPU clock speed in MHz."""
        return self.clock_speed_mhz

    def getCpuL1Cache(self) -> int:
        """Get L1 cache size in KB."""
        return self.l1_cache_kb

    def getCpuL2Cache(self) -> int:
        """Get L2 cache size in KB."""
        return self.l2_cache_kb

    def getCpuL3Cache(self) -> int:
        """Get L3 cache size in KB."""
        return self.l3_cache_kb

    def __str__(self):
        return f"{self.model_name} ({self.cores} cores, {self.sockets} sockets, {self.ram_size_gb}GB RAM)"


@dataclass
class GpuInfo:
    """GPU information."""

    device_id: str = ""
    revision_id: str = ""
    product_name: str = "Unknown"
    vendor: str = "AMD"
    vram_size_gb: int = 0
    sys_clock_mhz: int = 0
    mem_clock_mhz: int = 0
    pci_address: str = ""
    vbios: str = "Unknown"
    partition_mode: str = "Unknown"
    xgmi_type: str = "Unknown"
    host_driver: str = "Unknown"
    target_graphics_version: str = "Unknown"  # e.g., 'gfx942', 'gfx1100'
    firmwares: List[Dict[str, str]] = field(default_factory=list)

    def __str__(self):
        return f"{self.product_name} (Device ID: {self.device_id}, VRAM: {self.vram_size_gb}GB)"


def _get_rocm_tool_path(tool_name: str) -> str:
    """Get full path to ROCm tool (amd-smi) using THEROCK_BIN_DIR.

    Args:
        tool_name: Name of the tool ('rocm-smi' or 'amd-smi')

    Returns:
        str: Full path to tool if THEROCK_BIN_DIR is set, otherwise just tool name
    """
    therock_bin_dir = os.getenv("THEROCK_BIN_DIR")
    if therock_bin_dir:
        return os.path.join(therock_bin_dir, tool_name)
    return tool_name


class HardwareDetector:
    """Simple hardware detector."""

    def __init__(self):
        """Initialize hardware detector."""
        self.cpu_info = None
        self.gpu_list = []

    def detect_all(self):
        """Detect all hardware (CPU and GPU)."""
        self.detect_cpu()
        self.detect_gpu()

    def detect_cpu(self) -> CpuInfo:
        """Detect CPU information (cross-platform).

        Returns:
            CpuInfo object
        """
        if platform.system() == "Windows":
            return self._detect_cpu_windows()
        else:
            return self._detect_cpu_linux()

    def _detect_cpu_linux(self) -> CpuInfo:
        """Detect CPU information from /proc/cpuinfo and lscpu (Linux).

        Returns:
            CpuInfo object
        """
        try:
            # Read /proc/cpuinfo
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()

            # Extract model name
            model_match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
            model_name = model_match.group(1).strip() if model_match else "Unknown"

            # Count cores (logical processors)
            cores = len(re.findall(r"^processor\s*:", cpuinfo, re.MULTILINE))

            # Count sockets
            physical_ids = set(re.findall(r"physical id\s*:\s*(\d+)", cpuinfo))
            sockets = len(physical_ids) if physical_ids else 1

            # Get CPU MHz (clock speed)
            clock_match = re.search(r"cpu MHz\s*:\s*([\d.]+)", cpuinfo)
            clock_speed_mhz = int(float(clock_match.group(1))) if clock_match else 0

            # Get cache sizes from first processor
            l1_cache_kb = 0
            l2_cache_kb = 0
            l3_cache_kb = 0
            numa_nodes = 1

            # Try to get cache info from lscpu if available
            try:
                lscpu_output = subprocess.check_output(["lscpu", "-B"], text=True)

                # L1d cache (data)
                l1d_match = re.search(
                    r"L1d cache:\s*(\d+)\s*([KMG]?)", lscpu_output, re.IGNORECASE
                )
                if l1d_match:
                    size = int(l1d_match.group(1))
                    unit = l1d_match.group(2).upper()
                    if unit == "M":
                        size *= 1024
                    elif unit == "G":
                        size *= 1024 * 1024
                    l1_cache_kb = size

                # L1i cache (instruction)
                l1i_match = re.search(
                    r"L1i cache:\s*(\d+)\s*([KMG]?)", lscpu_output, re.IGNORECASE
                )
                if l1i_match:
                    size = int(l1i_match.group(1))
                    unit = l1i_match.group(2).upper()
                    if unit == "M":
                        size *= 1024
                    elif unit == "G":
                        size *= 1024 * 1024
                    l1_cache_kb += size

                # L2 cache
                l2_match = re.search(
                    r"L2 cache:\s*(\d+)\s*([KMG]?)", lscpu_output, re.IGNORECASE
                )
                if l2_match:
                    size = int(l2_match.group(1))
                    unit = l2_match.group(2).upper()
                    if unit == "M":
                        size *= 1024
                    elif unit == "G":
                        size *= 1024 * 1024
                    l2_cache_kb = size

                # L3 cache
                l3_match = re.search(
                    r"L3 cache:\s*(\d+)\s*([KMG]?)", lscpu_output, re.IGNORECASE
                )
                if l3_match:
                    size = int(l3_match.group(1))
                    unit = l3_match.group(2).upper()
                    if unit == "M":
                        size *= 1024
                    elif unit == "G":
                        size *= 1024 * 1024
                    l3_cache_kb = size

                # Get NUMA nodes
                numa_match = re.search(r"NUMA node\(s\):\s*(\d+)", lscpu_output)
                numa_nodes = int(numa_match.group(1)) if numa_match else 1

            except Exception as e:
                # Fallback: try to get cache from /proc/cpuinfo
                cache_match = re.search(r"cache size\s*:\s*(\d+)\s*KB", cpuinfo)
                if cache_match:
                    # This is usually L2 or L3 cache
                    l2_cache_kb = int(cache_match.group(1))

            # Get RAM size from /proc/meminfo
            ram_size_gb = 0
            try:
                with open("/proc/meminfo", "r") as f:
                    meminfo = f.read()
                mem_match = re.search(r"MemTotal:\s*(\d+)\s*kB", meminfo)
                if mem_match:
                    # Convert KB to GB
                    ram_size_gb = int(mem_match.group(1)) // (1024 * 1024)
            except Exception:
                pass

            self.cpu_info = CpuInfo(
                model_name=model_name,
                cores=cores,
                sockets=sockets,
                ram_size_gb=ram_size_gb,
                numa_nodes=numa_nodes,
                clock_speed_mhz=clock_speed_mhz,
                l1_cache_kb=l1_cache_kb,
                l2_cache_kb=l2_cache_kb,
                l3_cache_kb=l3_cache_kb,
            )

        except Exception:
            self.cpu_info = CpuInfo()

        return self.cpu_info

    def _detect_cpu_windows(self) -> CpuInfo:
        """Detect CPU information using Windows APIs and wmic (Windows).

        Returns:
            CpuInfo object
        """
        try:
            # Get CPU cores using multiprocessing
            cores = multiprocessing.cpu_count()

            # Get CPU info using wmic
            model_name = "Unknown"
            clock_speed_mhz = 0
            sockets = 1
            l1_cache_kb = 0
            l2_cache_kb = 0
            l3_cache_kb = 0
            ram_size_gb = 0

            try:
                # Get CPU model name
                cpu_output = subprocess.check_output(
                    ["wmic", "cpu", "get", "Name"], text=True, stderr=subprocess.DEVNULL
                ).strip()
                lines = [
                    line.strip() for line in cpu_output.split("\n") if line.strip()
                ]
                if len(lines) > 1:
                    model_name = lines[1]  # First line is header "Name"

                # Get CPU clock speed (in MHz)
                speed_output = subprocess.check_output(
                    ["wmic", "cpu", "get", "MaxClockSpeed"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                lines = [
                    line.strip() for line in speed_output.split("\n") if line.strip()
                ]
                if len(lines) > 1 and lines[1].isdigit():
                    clock_speed_mhz = int(lines[1])

                # Get number of physical CPUs (sockets)
                socket_output = subprocess.check_output(
                    ["wmic", "cpu", "get", "NumberOfCores"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                lines = [
                    line.strip() for line in socket_output.split("\n") if line.strip()
                ]
                if len(lines) > 1 and lines[1].isdigit():
                    physical_cores = int(lines[1])
                    # If we have hyperthreading, sockets = logical cores / (physical cores * 2)
                    # Otherwise sockets = logical cores / physical cores
                    if physical_cores > 0:
                        sockets = max(1, cores // physical_cores)

                # Get L2 cache size
                cache_output = subprocess.check_output(
                    ["wmic", "cpu", "get", "L2CacheSize"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                lines = [
                    line.strip() for line in cache_output.split("\n") if line.strip()
                ]
                if len(lines) > 1 and lines[1].isdigit():
                    l2_cache_kb = int(lines[1])

                # Get L3 cache size
                cache_output = subprocess.check_output(
                    ["wmic", "cpu", "get", "L3CacheSize"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                lines = [
                    line.strip() for line in cache_output.split("\n") if line.strip()
                ]
                if len(lines) > 1 and lines[1].isdigit():
                    l3_cache_kb = int(lines[1])

                # Get total RAM size (in KB, convert to GB)
                mem_output = subprocess.check_output(
                    ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                lines = [
                    line.strip() for line in mem_output.split("\n") if line.strip()
                ]
                if len(lines) > 1 and lines[1].isdigit():
                    ram_size_gb = int(lines[1]) // (1024 * 1024 * 1024)

            except Exception:
                # If wmic fails, use basic detection
                pass

            self.cpu_info = CpuInfo(
                model_name=model_name,
                cores=cores,
                sockets=sockets,
                ram_size_gb=ram_size_gb,
                numa_nodes=1,  # NUMA detection on Windows is complex, default to 1
                clock_speed_mhz=clock_speed_mhz,
                l1_cache_kb=l1_cache_kb,
                l2_cache_kb=l2_cache_kb,
                l3_cache_kb=l3_cache_kb,
            )

        except Exception:
            # Fallback to minimal detection with at least the core count
            try:
                cores = multiprocessing.cpu_count()
                self.cpu_info = CpuInfo(cores=cores, sockets=1)
            except Exception:
                self.cpu_info = CpuInfo()

        return self.cpu_info

    def detect_gpu(self) -> List[GpuInfo]:
        """Detect GPU information using ROCm tools and lspci.

        Detection hierarchy:
        1. amd-smi (primary) - Best for modern compute GPUs, SR-IOV, containers
        2. rocminfo (secondary) - Works with older ROCm versions, runtime-based
        3. lspci (fallback) - Basic detection, no ROCm required

        Returns:
            List of GpuInfo objects
        """
        self.gpu_list = []

        # PRIMARY METHOD: Try amd-smi first (modern, preferred)
        log.debug("Trying primary GPU detection with amd-smi...")
        if self._detect_gpu_with_amd_smi():
            log.debug(f"Successfully detected {len(self.gpu_list)} GPU(s) with amd-smi")
            return self.gpu_list

        # SECONDARY METHOD: Try rocminfo (older ROCm versions, runtime-based)
        log.debug("Primary detection found 0 GPUs, trying rocminfo method...")
        if self._detect_gpu_with_rocminfo():
            log.debug(
                f"Successfully detected {len(self.gpu_list)} GPU(s) with rocminfo"
            )
            return self.gpu_list

        # FALLBACK METHOD: Use lspci (works without ROCm)
        log.debug("Secondary detection found 0 GPUs, trying fallback lspci method...")
        self._detect_gpu_with_lspci()

        if len(self.gpu_list) > 0:
            log.debug(f"Successfully detected {len(self.gpu_list)} GPU(s) with lspci")
        else:
            log.debug("No GPUs detected by any method")

        return self.gpu_list

    def _detect_gpu_with_lspci(self) -> bool:
        """Detect GPUs using lspci (fallback method).

        Returns:
            True if GPUs detected, False otherwise
        """
        try:
            # Run lspci to find AMD GPUs
            log.debug("Running lspci to detect AMD GPUs...")
            result = subprocess.run(
                ["lspci", "-d", "1002:", "-nn"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                log.debug(f"lspci failed with return code {result.returncode}")
                return False

            log.debug(f"lspci output:\n{result.stdout}")

            # Parse output
            for line in result.stdout.splitlines():
                # Only actual GPUs (VGA/Display controller)
                if "VGA compatible controller" in line or "Display controller" in line:
                    # Extract PCI address (XX:XX.X)
                    pci_match = re.match(
                        r"^([0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])", line
                    )
                    pci_address = pci_match.group(1) if pci_match else ""

                    # Extract device ID from [1002:XXXX]
                    device_id_match = re.search(r"\[1002:([0-9a-fA-F]{4})\]", line)
                    device_id = device_id_match.group(1) if device_id_match else ""

                    # Extract product name
                    parts = line.split("]:")
                    if len(parts) >= 2:
                        product_part = parts[-1].strip()
                        product_name = re.sub(r"\s*\([^)]*\)\s*$", "", product_part)
                        product_name = re.sub(r"\s*\[[^\]]*\]\s*$", "", product_name)
                    else:
                        product_name = "AMD GPU"

                    # Get detailed info from lspci -v
                    revision_id = ""
                    vram_size_gb = 0

                    if pci_address:
                        try:
                            detail_result = subprocess.run(
                                ["lspci", "-s", pci_address, "-vv"],
                                capture_output=True,
                                text=True,
                                timeout=5,
                            )
                            detail_output = detail_result.stdout

                            # Extract revision ID (multiple formats)
                            # Format 1: "rev 0a" or "rev a1"
                            rev_match = re.search(
                                r"\(rev\s+([0-9a-fA-F]{2})\)",
                                detail_output,
                                re.IGNORECASE,
                            )
                            if rev_match:
                                revision_id = rev_match.group(1)
                            else:
                                # Format 2: "Revision: 0a"
                                rev_match = re.search(
                                    r"Revision:\s*([0-9a-fA-F]{2})",
                                    detail_output,
                                    re.IGNORECASE,
                                )
                                if rev_match:
                                    revision_id = rev_match.group(1)

                            # Also try from the main lspci line (appears after device ID)
                            if not revision_id:
                                rev_match = re.search(
                                    r"\[1002:[0-9a-fA-F]{4}\].*?\(rev\s+([0-9a-fA-F]{2})\)",
                                    line,
                                    re.IGNORECASE,
                                )
                                if rev_match:
                                    revision_id = rev_match.group(1)

                            log.debug(f"GPU {pci_address}: revision_id={revision_id}")

                            # Extract VRAM from memory regions
                            # Look for large memory regions (typically VRAM)
                            mem_regions = re.findall(
                                r"Memory at [0-9a-f]+ \(.*?\) \[size=(\d+)([MGT])\]",
                                detail_output,
                                re.IGNORECASE,
                            )
                            if mem_regions:
                                max_size = 0
                                for size_str, unit in mem_regions:
                                    size = int(size_str)
                                    if unit.upper() == "G":
                                        size_gb = size
                                    elif unit.upper() == "M":
                                        size_gb = size / 1024
                                    elif unit.upper() == "T":
                                        size_gb = size * 1024
                                    else:
                                        size_gb = 0

                                    if size_gb > max_size:
                                        max_size = size_gb

                                vram_size_gb = int(max_size)

                            log.debug(f"GPU {pci_address}: vram_size_gb={vram_size_gb}")

                        except Exception as e:
                            log.debug(
                                f"Error getting GPU details for {pci_address}: {e}"
                            )

                    gpu = GpuInfo(
                        device_id=device_id,
                        revision_id=revision_id,
                        product_name=product_name.strip(),
                        vendor="AMD",
                        vram_size_gb=vram_size_gb,
                        pci_address=pci_address,
                    )
                    self.gpu_list.append(gpu)

            return len(self.gpu_list) > 0

        except Exception as e:
            log.debug(f"lspci detection failed: {e}")
            return False

    def _detect_gpu_with_rocminfo(self) -> bool:
        """Detect GPUs using rocminfo (secondary method, ROCm runtime-based).

        Returns:
            True if GPUs detected, False otherwise
        """
        try:
            # Try to find rocminfo
            rocminfo_cmd = _get_rocm_tool_path("rocminfo")
            log.debug(f"Running {rocminfo_cmd} to detect AMD GPUs...")

            result = subprocess.run(
                [rocminfo_cmd],
                capture_output=True,
                text=True,
                timeout=15,
            )

            if result.returncode != 0:
                log.debug(f"rocminfo failed with return code {result.returncode}")
                return False

            log.debug(f"rocminfo output length: {len(result.stdout)} characters")

            # Parse rocminfo output
            # rocminfo outputs sections for each agent (GPU)
            # Format:
            # Agent 1                  *******
            #   Name:                    gfx942
            #   Marketing Name:          AMD Instinct MI300X
            #   Vendor Name:             AMD
            #   Device Type:             GPU
            #   ...

            current_agent = None
            gpu_agents = []

            for line in result.stdout.splitlines():
                # Detect start of agent section
                if line.strip().startswith("Agent ") and "*" in line:
                    # Save previous agent if it was a GPU
                    if current_agent and current_agent.get("device_type") == "GPU":
                        gpu_agents.append(current_agent)

                    # Start new agent
                    agent_match = re.match(r"Agent\s+(\d+)", line.strip())
                    if agent_match:
                        current_agent = {"agent_id": agent_match.group(1)}

                # Parse agent properties
                elif current_agent is not None and ":" in line:
                    key_value = line.split(":", 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = key_value[1].strip()

                        # Map rocminfo fields to our properties
                        if key == "Name":
                            current_agent["gfx_version"] = value
                        elif key == "Marketing Name":
                            current_agent["marketing_name"] = value
                        elif key == "Vendor Name":
                            current_agent["vendor"] = value
                        elif key == "Device Type":
                            current_agent["device_type"] = value
                        elif key == "Uuid":
                            current_agent["uuid"] = value
                        elif key == "Location (Bus/Device/Function)":
                            # Format: "Location (Bus/Device/Function): 0000:1b:00.0"
                            # Convert to XX:XX.X format
                            location_match = re.search(
                                r"([0-9a-fA-F]{4}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-9a-fA-F])",
                                value,
                            )
                            if location_match:
                                # Extract just the bus:device.function part (skip domain)
                                pci_addr = f"{location_match.group(2)}:{location_match.group(3)}.{location_match.group(4)}"
                                current_agent["pci_address"] = pci_addr

            # Don't forget the last agent
            if current_agent and current_agent.get("device_type") == "GPU":
                gpu_agents.append(current_agent)

            log.debug(f"Found {len(gpu_agents)} GPU agent(s) in rocminfo output")

            # Convert to GpuInfo objects
            for agent in gpu_agents:
                gpu_info = GpuInfo(
                    vendor=agent.get("vendor", "AMD"),
                    product_name=agent.get("marketing_name", "AMD GPU"),
                    target_graphics_version=agent.get("gfx_version", "Unknown"),
                    pci_address=agent.get("pci_address", ""),
                )
                self.gpu_list.append(gpu_info)

            return len(self.gpu_list) > 0

        except FileNotFoundError:
            log.debug("rocminfo command not found")
            return False
        except Exception as e:
            log.debug(f"rocminfo detection error: {e}")
            return False

    def _detect_gpu_with_amd_smi(self) -> bool:
        """Detect GPUs using amd-smi (primary method for compute GPUs).

        Returns:
            True if GPUs detected, False otherwise
        """
        try:
            amd_smi_cmd = _get_rocm_tool_path("amd-smi")
            log.debug(f"Running {amd_smi_cmd} static --json for GPU detection...")

            result = subprocess.run(
                [amd_smi_cmd, "static", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                log.debug(f"amd-smi not available (exit code {result.returncode})")
                return False

            log.debug(f"amd-smi static output: {result.stdout[:200]}...")
            data = json.loads(result.stdout)

            # Handle different amd-smi output formats
            gpu_list_data = None
            if isinstance(data, dict) and "gpu_data" in data:
                # New format: {"gpu_data": [...]}
                gpu_list_data = data["gpu_data"]
                log.debug("Detected new amd-smi format (gpu_data wrapper)")
            elif isinstance(data, list):
                # Old format: [...]
                gpu_list_data = data
                log.debug("Detected old amd-smi format (direct list)")
            else:
                log.debug(f"Unexpected amd-smi output format: {type(data)}")
                return False

            if not isinstance(gpu_list_data, list) or len(gpu_list_data) == 0:
                log.debug("No GPUs found in amd-smi output")
                return False

            # Parse GPU data and create GpuInfo objects
            for gpu_data in gpu_list_data:
                gpu_info = GpuInfo(vendor="AMD")

                # Extract asic information
                if "asic" in gpu_data:
                    asic = gpu_data["asic"]
                    if isinstance(asic, dict):
                        gpu_info.product_name = asic.get("market_name", "AMD GPU")
                        gpu_info.device_id = asic.get("device_id", "").replace("0x", "")
                        gpu_info.revision_id = asic.get("rev_id", "").replace("0x", "")
                        gpu_info.target_graphics_version = asic.get(
                            "target_graphics_version", "Unknown"
                        )

                # Extract bus information
                if "bus" in gpu_data:
                    bus = gpu_data["bus"]
                    if isinstance(bus, dict):
                        gpu_info.pci_address = bus.get("bdf", "").replace("0000:", "")

                # Extract VRAM information
                if "vram" in gpu_data:
                    vram = gpu_data["vram"]
                    if isinstance(vram, dict):
                        vram_mb = vram.get("total", 0)
                        gpu_info.vram_size_gb = vram_mb // 1024 if vram_mb > 0 else 0

                # Extract VBIOS
                if "vbios" in gpu_data:
                    gpu_info.vbios = str(gpu_data.get("vbios", "Unknown"))

                # Extract partition mode
                if "partition" in gpu_data:
                    gpu_info.partition_mode = str(gpu_data.get("partition", "Unknown"))

                # Extract driver info
                if "driver" in gpu_data:
                    gpu_info.host_driver = str(gpu_data.get("driver", "Unknown"))
                elif "driver_version" in gpu_data:
                    gpu_info.host_driver = str(
                        gpu_data.get("driver_version", "Unknown")
                    )

                self.gpu_list.append(gpu_info)

            return len(self.gpu_list) > 0

        except FileNotFoundError:
            log.debug("amd-smi command not found")
            return False
        except json.JSONDecodeError as e:
            log.debug(f"Failed to parse amd-smi JSON output: {e}")
            return False
        except Exception as e:
            log.debug(f"amd-smi detection error: {e}")
            return False

    @staticmethod
    def get_gpu_info_from_amd_smi() -> Tuple[int, Optional[str]]:
        """Get GPU count and target graphics version directly from amd-smi.

        Returns:
            Tuple[int, Optional[str]]: (gpu_count, target_graphics_version)
                - gpu_count: Number of GPUs detected (0 if detection fails)
                - target_graphics_version: GPU architecture (e.g., 'gfx942', 'gfx1100') or None
        """
        try:
            amd_smi_cmd = _get_rocm_tool_path("amd-smi")
            log.debug(f"Trying {amd_smi_cmd} static --json for GPU count...")
            result = subprocess.run(
                [amd_smi_cmd, "static", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                log.debug(f"amd-smi not available (exit code {result.returncode})")
                return 0, None

            log.debug(f"amd-smi static output: {result.stdout[:200]}...")
            data = json.loads(result.stdout)

            # Handle different amd-smi output formats
            gpu_list = None
            if isinstance(data, dict) and "gpu_data" in data:
                # New format: {"gpu_data": [...]}
                gpu_list = data["gpu_data"]
                log.debug("Detected new amd-smi format (gpu_data wrapper)")
            elif isinstance(data, list):
                # Old format: [...]
                gpu_list = data
                log.debug("Detected old amd-smi format (direct list)")
            else:
                log.debug(f"Unexpected amd-smi output format: {type(data)}")
                return 0, None

            if not isinstance(gpu_list, list):
                log.debug("gpu_list is not a list")
                return 0, None

            gpu_count = len(gpu_list)

            # Extract target_graphics_version from first GPU
            target_graphics_version = None
            if gpu_count > 0:
                first_gpu = gpu_list[0]

                # Try different possible field names for gfx version
                # Common fields: asic.target_graphics_version, asic.market_name, asic
                if "asic" in first_gpu:
                    asic_data = first_gpu["asic"]
                    if isinstance(asic_data, dict):
                        # Try target_graphics_version field
                        target_graphics_version = asic_data.get(
                            "target_graphics_version"
                        )

                        # Try market_name which might contain gfx info
                        if not target_graphics_version:
                            market_name = asic_data.get("market_name", "")
                            # Extract gfxXXXX pattern from market name
                            gfx_match = re.search(r"gfx\w+", market_name, re.IGNORECASE)
                            if gfx_match:
                                target_graphics_version = gfx_match.group(0).lower()

                # Try top-level fields
                if not target_graphics_version:
                    target_graphics_version = first_gpu.get("target_graphics_version")

                if not target_graphics_version:
                    target_graphics_version = first_gpu.get("gfx_version")

            log.debug(
                f"Detected {gpu_count} GPU(s), target_graphics_version={target_graphics_version}"
            )
            return gpu_count, target_graphics_version

        except FileNotFoundError:
            log.debug("amd-smi command not found")
            return 0, None
        except json.JSONDecodeError as e:
            log.debug(f"Failed to parse amd-smi JSON output: {e}")
            return 0, None
        except Exception as e:
            log.debug(f"amd-smi detection error: {e}")
            return 0, None

    def get_cpu(self) -> Optional[CpuInfo]:
        """Get detected CPU information.

        Returns:
            CpuInfo: Detected CPU info or None
        """
        return self.cpu_info

    def get_is_cpu_initialized(self) -> bool:
        """Check if CPU detection completed.

        Returns:
            bool: True if CPU info available
        """
        return self.cpu_info is not None

    def get_is_gpu_initialized(self) -> bool:
        """Check if GPU detection completed.

        Returns:
            bool: True if GPU detection was attempted
        """
        return True  # Always true after detect_all() is called

    def getGpu(self):
        """Get GPU handler (camelCase compatibility alias).

        Returns:
            Self for accessing .adapters attribute
        """
        return self

    @property
    def adapters(self) -> List[GpuInfo]:
        """Get GPU adapters list (compatibility property).

        Returns:
            List[GpuInfo]: List of detected GPUs
        """
        return self.gpu_list
