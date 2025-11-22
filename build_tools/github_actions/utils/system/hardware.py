"""Hardware detection for CPU and GPU using system commands and ROCm tools."""

import subprocess
import re
import os
from typing import List, Optional, Dict
from dataclasses import dataclass, field


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
    firmwares: List[Dict[str, str]] = field(default_factory=list)
    
    def __str__(self):
        return f"{self.product_name} (Device ID: {self.device_id}, VRAM: {self.vram_size_gb}GB)"


def _get_rocm_tool_path(tool_name: str) -> str:
    """Get full path to ROCm tool (rocm-smi or amd-smi) using THEROCK_BIN_DIR.
    
    Args:
        tool_name: Name of the tool ('rocm-smi' or 'amd-smi')
    
    Returns:
        str: Full path to tool if THEROCK_BIN_DIR is set, otherwise just tool name
    """
    therock_bin_dir = os.getenv('THEROCK_BIN_DIR')
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
        """Detect CPU information from /proc/cpuinfo and lscpu.
        
        Returns:
            CpuInfo object
        """
        try:
            # Read /proc/cpuinfo
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
            
            # Extract model name
            model_match = re.search(r'model name\s*:\s*(.+)', cpuinfo)
            model_name = model_match.group(1).strip() if model_match else "Unknown"
            
            # Count cores (logical processors)
            cores = len(re.findall(r'^processor\s*:', cpuinfo, re.MULTILINE))
            
            # Count sockets
            physical_ids = set(re.findall(r'physical id\s*:\s*(\d+)', cpuinfo))
            sockets = len(physical_ids) if physical_ids else 1
            
            # Get CPU MHz (clock speed)
            clock_match = re.search(r'cpu MHz\s*:\s*([\d.]+)', cpuinfo)
            clock_speed_mhz = int(float(clock_match.group(1))) if clock_match else 0
            
            # Get cache sizes from first processor
            l1_cache_kb = 0
            l2_cache_kb = 0
            l3_cache_kb = 0
            numa_nodes = 1
            
            # Try to get cache info from lscpu if available
            try:
                lscpu_output = subprocess.check_output(['lscpu', '-B'], text=True)
                
                # L1d cache (data)
                l1d_match = re.search(r'L1d cache:\s*(\d+)\s*([KMG]?)', lscpu_output, re.IGNORECASE)
                if l1d_match:
                    size = int(l1d_match.group(1))
                    unit = l1d_match.group(2).upper()
                    if unit == 'M':
                        size *= 1024
                    elif unit == 'G':
                        size *= 1024 * 1024
                    l1_cache_kb = size
                
                # L1i cache (instruction)
                l1i_match = re.search(r'L1i cache:\s*(\d+)\s*([KMG]?)', lscpu_output, re.IGNORECASE)
                if l1i_match:
                    size = int(l1i_match.group(1))
                    unit = l1i_match.group(2).upper()
                    if unit == 'M':
                        size *= 1024
                    elif unit == 'G':
                        size *= 1024 * 1024
                    l1_cache_kb += size
                
                # L2 cache
                l2_match = re.search(r'L2 cache:\s*(\d+)\s*([KMG]?)', lscpu_output, re.IGNORECASE)
                if l2_match:
                    size = int(l2_match.group(1))
                    unit = l2_match.group(2).upper()
                    if unit == 'M':
                        size *= 1024
                    elif unit == 'G':
                        size *= 1024 * 1024
                    l2_cache_kb = size
                
                # L3 cache
                l3_match = re.search(r'L3 cache:\s*(\d+)\s*([KMG]?)', lscpu_output, re.IGNORECASE)
                if l3_match:
                    size = int(l3_match.group(1))
                    unit = l3_match.group(2).upper()
                    if unit == 'M':
                        size *= 1024
                    elif unit == 'G':
                        size *= 1024 * 1024
                    l3_cache_kb = size
                
                # Get NUMA nodes
                numa_match = re.search(r'NUMA node\(s\):\s*(\d+)', lscpu_output)
                numa_nodes = int(numa_match.group(1)) if numa_match else 1
                
            except Exception as e:
                # Fallback: try to get cache from /proc/cpuinfo
                cache_match = re.search(r'cache size\s*:\s*(\d+)\s*KB', cpuinfo)
                if cache_match:
                    # This is usually L2 or L3 cache
                    l2_cache_kb = int(cache_match.group(1))
            
            # Get RAM size from /proc/meminfo
            ram_size_gb = 0
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                mem_match = re.search(r'MemTotal:\s*(\d+)\s*kB', meminfo)
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
                l3_cache_kb=l3_cache_kb
            )
            
        except Exception:
            self.cpu_info = CpuInfo()
        
        return self.cpu_info
    
    def detect_gpu(self) -> List[GpuInfo]:
        """Detect GPU information using lspci and ROCm tools.
        
        Returns:
            List of GpuInfo objects
        """
        self.gpu_list = []
        
        try:
            import logging
            logger = logging.getLogger(__name__)
            
            # Run lspci to find AMD GPUs
            logger.debug("Running lspci to detect AMD GPUs...")
            result = subprocess.run(
                ['lspci', '-d', '1002:', '-nn'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.debug(f"lspci failed with return code {result.returncode}")
                return self.gpu_list
            
            logger.debug(f"lspci output:\n{result.stdout}")
            
            # Parse output
            for line in result.stdout.splitlines():
                # Only actual GPUs (VGA/Display controller)
                if 'VGA compatible controller' in line or 'Display controller' in line:
                    # Extract PCI address (XX:XX.X)
                    pci_match = re.match(r'^([0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])', line)
                    pci_address = pci_match.group(1) if pci_match else ""
                    
                    # Extract device ID from [1002:XXXX]
                    device_id_match = re.search(r'\[1002:([0-9a-fA-F]{4})\]', line)
                    device_id = device_id_match.group(1) if device_id_match else ""
                    
                    # Extract product name
                    parts = line.split(']:')
                    if len(parts) >= 2:
                        product_part = parts[-1].strip()
                        product_name = re.sub(r'\s*\([^)]*\)\s*$', '', product_part)
                        product_name = re.sub(r'\s*\[[^\]]*\]\s*$', '', product_name)
                    else:
                        product_name = "AMD GPU"
                    
                    # Get detailed info from lspci -v
                    revision_id = ""
                    vram_size_gb = 0
                    
                    if pci_address:
                        try:
                            detail_result = subprocess.run(
                                ['lspci', '-s', pci_address, '-vv'],
                                capture_output=True,
                                text=True,
                                timeout=5
                            )
                            detail_output = detail_result.stdout
                            
                            # Extract revision ID (multiple formats)
                            # Format 1: "rev 0a" or "rev a1"
                            rev_match = re.search(r'\(rev\s+([0-9a-fA-F]{2})\)', detail_output, re.IGNORECASE)
                            if rev_match:
                                revision_id = rev_match.group(1)
                            else:
                                # Format 2: "Revision: 0a"
                                rev_match = re.search(r'Revision:\s*([0-9a-fA-F]{2})', detail_output, re.IGNORECASE)
                                if rev_match:
                                    revision_id = rev_match.group(1)
                            
                            # Also try from the main lspci line (appears after device ID)
                            if not revision_id:
                                rev_match = re.search(r'\[1002:[0-9a-fA-F]{4}\].*?\(rev\s+([0-9a-fA-F]{2})\)', line, re.IGNORECASE)
                                if rev_match:
                                    revision_id = rev_match.group(1)
                            
                            logger.debug(f"GPU {pci_address}: revision_id={revision_id}")
                            
                            # Extract VRAM from memory regions
                            # Look for large memory regions (typically VRAM)
                            mem_regions = re.findall(r'Memory at [0-9a-f]+ \(.*?\) \[size=(\d+)([MGT])\]', detail_output, re.IGNORECASE)
                            if mem_regions:
                                max_size = 0
                                for size_str, unit in mem_regions:
                                    size = int(size_str)
                                    if unit.upper() == 'G':
                                        size_gb = size
                                    elif unit.upper() == 'M':
                                        size_gb = size / 1024
                                    elif unit.upper() == 'T':
                                        size_gb = size * 1024
                                    else:
                                        size_gb = 0
                                    
                                    if size_gb > max_size:
                                        max_size = size_gb
                                
                                vram_size_gb = int(max_size)
                            
                            logger.debug(f"GPU {pci_address}: vram_size_gb={vram_size_gb}")
                        
                        except Exception as e:
                            logger.debug(f"Error getting GPU details for {pci_address}: {e}")
                    
                    gpu = GpuInfo(
                        device_id=device_id,
                        revision_id=revision_id,
                        product_name=product_name.strip(),
                        vendor="AMD",
                        vram_size_gb=vram_size_gb,
                        pci_address=pci_address
                    )
                    self.gpu_list.append(gpu)
            
            # Try to enhance with rocm-smi or amd-smi for clocks
            self._enhance_gpu_with_rocm()
            
        except Exception:
            pass
        
        return self.gpu_list
    
    def _enhance_gpu_with_rocm(self):
        """Enhance GPU info with ROCm tools (amd-smi or rocm-smi) for VRAM, clocks, and firmware."""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.debug("Attempting to enhance GPU info with ROCm tools...")
        
        # Try amd-smi first (newer)
        if self._try_amd_smi():
            logger.debug("Successfully enhanced GPU info with amd-smi")
            return
        
        # Fallback to rocm-smi
        if self._try_rocm_smi():
            logger.debug("Successfully enhanced GPU info with rocm-smi")
        else:
            logger.debug("No ROCm tools available for GPU enhancement")
    
    def _try_amd_smi(self) -> bool:
        """Try to get GPU info from amd-smi."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            amd_smi_cmd = _get_rocm_tool_path('amd-smi')
            logger.debug(f"Trying {amd_smi_cmd} static --json...")
            result = subprocess.run(
                [amd_smi_cmd, 'static', '--json'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.debug(f"amd-smi not available (exit code {result.returncode})")
                return False
            
            logger.debug(f"amd-smi static output: {result.stdout[:200]}...")
            
            import json
            data = json.loads(result.stdout)
            
            # Parse amd-smi output and update GPU info
            for i, gpu in enumerate(self.gpu_list):
                if i < len(data):
                    gpu_data = data[i]
                    if 'vram' in gpu_data and gpu.vram_size_gb == 0:
                        vram_mb = gpu_data.get('vram', {}).get('total', 0)
                        gpu.vram_size_gb = vram_mb // 1024
                    
                    # Extract VBIOS version
                    if 'vbios' in gpu_data:
                        gpu.vbios = gpu_data.get('vbios', 'Unknown')
                    
                    # Extract partition mode
                    if 'partition' in gpu_data:
                        gpu.partition_mode = gpu_data.get('partition', 'Unknown')
                    
                    # Extract XGMI info
                    if 'xgmi' in gpu_data:
                        xgmi_info = gpu_data.get('xgmi', {})
                        if isinstance(xgmi_info, dict):
                            gpu.xgmi_type = xgmi_info.get('type', 'Unknown')
                        else:
                            gpu.xgmi_type = str(xgmi_info)
                    
                    # Extract driver info
                    if 'driver' in gpu_data:
                        gpu.host_driver = gpu_data.get('driver', 'Unknown')
                    elif 'driver_version' in gpu_data:
                        gpu.host_driver = gpu_data.get('driver_version', 'Unknown')
                    
                    # Extract firmware info
                    if 'firmware' in gpu_data:
                        firmware_data = gpu_data.get('firmware', {})
                        if isinstance(firmware_data, dict):
                            for fw_name, fw_version in firmware_data.items():
                                gpu.firmwares.append({
                                    'name': fw_name,
                                    'version': str(fw_version)
                                })
                        elif isinstance(firmware_data, list):
                            for fw_item in firmware_data:
                                if isinstance(fw_item, dict):
                                    gpu.firmwares.append({
                                        'name': fw_item.get('name', 'Unknown'),
                                        'version': fw_item.get('version', 'Unknown')
                                    })
            
            # Get clocks
            logger.debug(f"Trying {amd_smi_cmd} metric --json...")
            result = subprocess.run(
                [amd_smi_cmd, 'metric', '--json'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.debug(f"amd-smi metric output: {result.stdout[:200]}...")
                data = json.loads(result.stdout)
                for i, gpu in enumerate(self.gpu_list):
                    if i < len(data):
                        gpu_data = data[i]
                        clocks = gpu_data.get('clocks', {})
                        gpu.sys_clock_mhz = clocks.get('sclk', 0)
                        gpu.mem_clock_mhz = clocks.get('mclk', 0)
                        logger.debug(f"GPU {i}: sys_clock={gpu.sys_clock_mhz} MHz, mem_clock={gpu.mem_clock_mhz} MHz")
            else:
                logger.debug(f"amd-smi metric failed (exit code {result.returncode})")
            
            return True
            
        except FileNotFoundError:
            logger.debug("amd-smi command not found")
            return False
        except Exception as e:
            logger.debug(f"amd-smi error: {e}")
            return False
    
    def _try_rocm_smi(self) -> bool:
        """Try to get GPU info from rocm-smi."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            rocm_smi_cmd = _get_rocm_tool_path('rocm-smi')
            logger.debug(f"Trying {rocm_smi_cmd} --showmeminfo vram --json...")
            result = subprocess.run(
                [rocm_smi_cmd, '--showmeminfo', 'vram', '--json'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.debug(f"rocm-smi not available (exit code {result.returncode})")
                return False
            
            logger.debug(f"rocm-smi meminfo output: {result.stdout[:200]}...")
            
            import json
            data = json.loads(result.stdout)
            
            # Parse rocm-smi output
            for i, gpu in enumerate(self.gpu_list):
                gpu_key = f"card{i}"
                if gpu_key in data and gpu.vram_size_gb == 0:
                    vram_mb = data[gpu_key].get('VRAM Total Memory (B)', 0) // (1024 * 1024)
                    gpu.vram_size_gb = vram_mb // 1024
            
            # Try to get VBIOS and other info from rocm-smi
            try:
                result = subprocess.run(
                    [rocm_smi_cmd, '--showvbios', '--json'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    vbios_data = json.loads(result.stdout)
                    for i, gpu in enumerate(self.gpu_list):
                        gpu_key = f"card{i}"
                        if gpu_key in vbios_data:
                            gpu.vbios = vbios_data[gpu_key].get('VBIOS Version', 'Unknown')
                            logger.debug(f"GPU {i}: vbios = {gpu.vbios}")
            except Exception as e:
                logger.debug(f"Failed to get VBIOS info: {e}")
            
            # Try to get firmware info from rocm-smi
            try:
                result = subprocess.run(
                    [rocm_smi_cmd, '--showfwinfo', '--json'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    fw_data = json.loads(result.stdout)
                    for i, gpu in enumerate(self.gpu_list):
                        gpu_key = f"card{i}"
                        if gpu_key in fw_data:
                            fw_info = fw_data[gpu_key]
                            # rocm-smi returns firmware info as dict
                            for fw_name, fw_version in fw_info.items():
                                if fw_name not in ['card', 'GPU ID']:  # Skip metadata
                                    gpu.firmwares.append({
                                        'name': fw_name,
                                        'version': str(fw_version)
                                    })
                            logger.debug(f"GPU {i}: found {len(gpu.firmwares)} firmwares")
            except Exception as e:
                logger.debug(f"Failed to get firmware info: {e}")
            
            # Get clocks
            logger.debug(f"Trying {rocm_smi_cmd} --showclocks --json...")
            result = subprocess.run(
                [rocm_smi_cmd, '--showclocks', '--json'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.debug(f"rocm-smi clocks output: {result.stdout[:200]}...")
                data = json.loads(result.stdout)
                for i, gpu in enumerate(self.gpu_list):
                    gpu_key = f"card{i}"
                    if gpu_key in data:
                        gpu_data = data[gpu_key]
                        # Extract current clocks
                        sclk = gpu_data.get('sclk', {}).get('level', [])
                        mclk = gpu_data.get('mclk', {}).get('level', [])
                        
                        if sclk:
                            # Get current clock (marked with *)
                            for clk in sclk:
                                if '*' in str(clk):
                                    clock_match = re.search(r'(\d+)Mhz', str(clk), re.IGNORECASE)
                                    if clock_match:
                                        gpu.sys_clock_mhz = int(clock_match.group(1))
                                        logger.debug(f"GPU {i}: sys_clock from rocm-smi = {gpu.sys_clock_mhz} MHz")
                        
                        if mclk:
                            for clk in mclk:
                                if '*' in str(clk):
                                    clock_match = re.search(r'(\d+)Mhz', str(clk), re.IGNORECASE)
                                    if clock_match:
                                        gpu.mem_clock_mhz = int(clock_match.group(1))
                                        logger.debug(f"GPU {i}: mem_clock from rocm-smi = {gpu.mem_clock_mhz} MHz")
            else:
                logger.debug(f"rocm-smi --showclocks failed (exit code {result.returncode})")
            
            return True
            
        except FileNotFoundError:
            logger.debug("rocm-smi command not found")
            return False
        except Exception as e:
            logger.debug(f"rocm-smi error: {e}")
            return False
    
    def get_cpu(self) -> Optional[CpuInfo]:
        """Get detected CPU information.
        
        Returns:
            CpuInfo: Detected CPU info or None
        """
        return self.cpu_info
    
    def get_gpus(self) -> List[GpuInfo]:
        """Get detected GPU list.
        
        Returns:
            List[GpuInfo]: List of detected GPUs
        """
        return self.gpu_list
    
    def has_gpu(self) -> bool:
        """Check if any GPU is detected.
        
        Returns:
            bool: True if at least one GPU detected
        """
        return len(self.gpu_list) > 0
    
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
    
    def getCpu(self) -> Optional[CpuInfo]:
        """Get CPU information (camelCase compatibility alias).
        
        Returns:
            CpuInfo: Detected CPU info or None
        """
        return self.get_cpu()
    
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