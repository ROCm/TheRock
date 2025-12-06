"""System detector combining platform, hardware, and ROCm detection into unified context."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from ..logger import log
from .hardware import HardwareDetector
from .platform import PlatformDetector
from .rocm_detector import ROCmDetector


def format_memory_size(size_gb: int) -> str:
    """Format memory size with appropriate units (GB or TB).
    
    Args:
        size_gb: Memory size in GB
        
    Returns:
        str: Formatted string (e.g., "124 GB", "1.5 TB")
    """
    if size_gb == 0:
        return "0 GB"
    elif size_gb >= 1024:
        # Convert to TB if >= 1TB
        size_tb = size_gb / 1024
        return f"{size_tb:.1f} TB"
    elif size_gb < 1:
        # Convert to MB if < 1GB
        size_mb = size_gb * 1024
        return f"{size_mb:.0f} MB"
    else:
        return f"{size_gb} GB"


def format_cache_size(size_kb: int) -> str:
    """Format cache size with appropriate units (KB, MB, or GB).
    
    Args:
        size_kb: Cache size in KB
        
    Returns:
        str: Formatted string (e.g., "256 KB", "8 MB")
    """
    if size_kb == 0:
        return "0 KB"
    elif size_kb >= 1024:
        # Convert to MB if >= 1MB
        size_mb = size_kb / 1024
        if size_mb >= 1024:
            # Convert to GB if >= 1GB
            size_gb = size_mb / 1024
            return f"{size_gb:.1f} GB"
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_kb} KB"


def format_clock_speed(speed_mhz: int) -> str:
    """Format clock speed with appropriate units (MHz or GHz).
    
    Args:
        speed_mhz: Clock speed in MHz
        
    Returns:
        str: Formatted string (e.g., "3500 MHz", "3.5 GHz")
    """
    if speed_mhz == 0:
        return "0 MHz"
    elif speed_mhz >= 1000:
        # Convert to GHz if >= 1GHz
        speed_ghz = speed_mhz / 1000
        return f"{speed_ghz:.2f} GHz"
    else:
        return f"{speed_mhz} MHz"


@dataclass
class SystemContext:
    """System context dataclass containing platform, CPU, GPU, and ROCm information."""
    
    # Platform info
    os_name: str
    os_version: str
    kernel: str
    hostname: str
    system_ip: str
    sbios: str
    
    # CPU info
    cpu_model: str
    cpu_cores: int
    cpu_sockets: int
    cpu_ram_size: int
    cpu_numa_nodes: int
    cpu_clock_speed: int
    cpu_l1_cache: int
    cpu_l2_cache: int
    cpu_l3_cache: int
    
    # GPU info
    gpu_count: int
    gpu_name: str
    gpu_marketing_name: str
    gpu_device_id: Optional[str]
    gpu_revision_id: Optional[str]
    gpu_vram_size: float
    gpu_sys_clock: int
    gpu_mem_clock: int
    gpu_vbios: str
    gpu_partition_mode: str
    gpu_xgmi_type: str
    gpu_host_driver: str
    gpu_firmwares: List[Dict[str, str]]
    gpu_devices: List[str]
    
    # ROCm info
    rocm_version: str
    rocm_build_type: str
    rocm_build_lib_type: str
    rocm_package_manager: str
    rocm_package_manager_version: str
    rocm_install_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system context to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary with all system information fields
        """
        return asdict(self)


class SystemDetector:
    """System detector for comprehensive platform, hardware, and ROCm detection.
    
    Example:
        >>> detector = SystemDetector()
        >>> context = detector.detect_all()
        >>> print(f"OS: {context.os_name}, GPU: {context.gpu_name}")
    """
    
    def __init__(self):
        """Initialize system detector with empty state."""
        self.platform_info = None
        self.hardware = None
        self.rocm_info = None
    
    def detect_all(self, verbose: bool = True) -> SystemContext:
        """Detect complete system information (platform, hardware, ROCm).
        
        Args:
            verbose: Log detection progress (default: True)
        
        Returns:
            SystemContext: Complete system information
        """
        if verbose:
            log.info("Detecting system information...")
        
        # Detect platform
        self.platform_info = PlatformDetector.detect()
        if verbose:
            log.debug(f"Platform: {self.platform_info.os_name} {self.platform_info.os_version}")
        
        # Detect hardware
        self.hardware = HardwareDetector()
        self.hardware.detect_all()
        if verbose:
            log.debug("Hardware detection complete")
        
        # Detect ROCm
        self.rocm_info = ROCmDetector.detect_rocm_info()
        if verbose:
            log.debug(f"ROCm: {self.rocm_info['rocm_version']}")
        
        # Build system context
        context = self.build_system_context()
        
        if verbose:
            log.info("✓ System detection complete")
        
        return context
    
    def build_system_context(self) -> SystemContext:
        """Build SystemContext dataclass from detected platform, hardware, and ROCm info.
        
        Returns:
            SystemContext: Complete system context
            
        Raises:
            RuntimeError: If detect_all() hasn't been called
        """
        if self.platform_info is None or self.hardware is None or self.rocm_info is None:
            raise RuntimeError(
                "System detection not complete. Call detect_all() first."
            )
        
        # Get CPU info
        cpu = self.hardware.get_cpu() if self.hardware.get_is_cpu_initialized() else None
        
        # Get GPU info
        gpu_devices = []
        if self.hardware.get_is_gpu_initialized():
            gpu = self.hardware.getGpu()
            gpu_devices = gpu.adapters if gpu.adapters else []
        
        return SystemContext(
            # Platform
            os_name=self.platform_info.os_name,
            os_version=self.platform_info.os_version,
            kernel=self.platform_info.kernel_release,
            hostname=self.platform_info.hostname,
            system_ip=PlatformDetector.get_system_ip(),
            sbios=self.platform_info.sbios,
            
            # CPU
            cpu_model=cpu.getCpuModelName() if cpu else 'Unknown',
            cpu_cores=cpu.getCpuCores() if cpu else 0,
            cpu_sockets=cpu.getCpuSockets() if cpu else 0,
            cpu_ram_size=cpu.getCpuRamSize() if cpu else 0,
            cpu_numa_nodes=cpu.getCpuNumaNodes() if cpu else 0,
            cpu_clock_speed=cpu.getCpuClockSpeed() if cpu else 0,
            cpu_l1_cache=cpu.getCpuL1Cache() if cpu else 0,
            cpu_l2_cache=cpu.getCpuL2Cache() if cpu else 0,
            cpu_l3_cache=cpu.getCpuL3Cache() if cpu else 0,
            
            # GPU
            gpu_count=len(gpu_devices),
            gpu_name=gpu_devices[0].product_name if gpu_devices else 'Unknown',
            gpu_marketing_name=gpu_devices[0].product_name if gpu_devices else 'Unknown',
            gpu_device_id=gpu_devices[0].device_id if gpu_devices else 'Unknown',
            gpu_revision_id=gpu_devices[0].revision_id if gpu_devices else 'Unknown',
            gpu_vram_size=gpu_devices[0].vram_size_gb if gpu_devices else 0,
            gpu_sys_clock=gpu_devices[0].sys_clock_mhz if gpu_devices else 0,
            gpu_mem_clock=gpu_devices[0].mem_clock_mhz if gpu_devices else 0,
            gpu_vbios=gpu_devices[0].vbios if gpu_devices else 'Unknown',
            gpu_partition_mode=gpu_devices[0].partition_mode if gpu_devices else 'Unknown',
            gpu_xgmi_type=gpu_devices[0].xgmi_type if gpu_devices else 'Unknown',
            gpu_host_driver=gpu_devices[0].host_driver if gpu_devices else 'Unknown',
            gpu_firmwares=gpu_devices[0].firmwares if gpu_devices else [],
            gpu_devices=[adapter.product_name for adapter in gpu_devices],
            
            # ROCm
            rocm_version=self.rocm_info['rocm_version'],
            rocm_build_type=self.rocm_info['rocm_build_type'],
            rocm_build_lib_type=self.rocm_info['rocm_build_lib_type'],
            rocm_package_manager=self.rocm_info['rocm_package_manager'],
            rocm_package_manager_version=self.rocm_info['rocm_package_manager_version'],
            rocm_install_type=self.rocm_info['install_type']
        )
    
    def print_system_summary(self, context: SystemContext):
        """Print formatted system information summary to console.
        
        Args:
            context: SystemContext with detected information
        """
        print("\n" + "=" * 70)
        print("SYSTEM INFORMATION")
        print("=" * 70)
        print(f"OS:           {context.os_name} {context.os_version}")
        print(f"Kernel:       {context.kernel}")
        print(f"Hostname:     {context.hostname}")
        print(f"System IP:    {context.system_ip}")
        print(f"SBIOS:        {context.sbios}")
        print()
        print(f"CPU:          {context.cpu_model}")
        print(f"  Cores:      {context.cpu_cores} (Sockets: {context.cpu_sockets})")
        print(f"  RAM:        {format_memory_size(context.cpu_ram_size)}")
        print(f"  Clock:      {format_clock_speed(context.cpu_clock_speed)}")
        print()
        print(f"GPU:          {context.gpu_name}")
        print(f"  Count:      {context.gpu_count}")
        print(f"  Device ID:  {context.gpu_device_id}")
        print(f"  VRAM:       {format_memory_size(int(context.gpu_vram_size))}")
        print(f"  Clocks:     {format_clock_speed(context.gpu_sys_clock)} / {format_clock_speed(context.gpu_mem_clock)}")
        print(f"  VBIOS:      {context.gpu_vbios}")
        print()
        print(f"ROCm:         {context.rocm_version}")
        print(f"  Build:      {context.rocm_build_type}")
        print(f"  Install:    {context.rocm_install_type}")
        print("=" * 70 + "\n")
    
    def log_system_info(self, context: SystemContext):
        """Log system information using logger with formatted output.
        
        Args:
            context: SystemContext with detected information
        """
        log.info(f"✓ Platform detected: {context.os_name} {context.os_version}")
        log.info(f"  Kernel: {context.kernel}")
        log.info(f"  Hostname: {context.hostname}")
        
        log.info(f"✓ CPU detected: {context.cpu_model}")
        log.info(f"  Cores: {context.cpu_cores}, Sockets: {context.cpu_sockets}")
        
        if context.gpu_count > 0:
            log.info(f"✓ GPU detected: {context.gpu_count} GPU(s)")
            log.info(f"  GPU 0: {context.gpu_name} (Device ID: {context.gpu_device_id})")
        else:
            log.warning("⚠ No GPU detected")
        
        log.info(f"✓ ROCm detected: {context.rocm_version}")
        log.info(f"  Install type: {context.rocm_install_type}")
        log.info(f"  Build type: {context.rocm_build_type}")