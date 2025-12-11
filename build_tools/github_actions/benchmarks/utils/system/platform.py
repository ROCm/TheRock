"""Platform detection for OS, kernel, hostname, IP, and BIOS information."""

import platform
import socket
from dataclasses import dataclass


@dataclass
class PlatformInfo:
    """Platform information dataclass with OS, kernel, hostname, and BIOS details."""
    os_name: str = "Unknown"
    os_version: str = "Unknown"
    kernel: str = "Unknown"
    hostname: str = "Unknown"
    architecture: str = "Unknown"
    sbios: str = "Unknown"
    
    @property
    def kernel_release(self) -> str:
        """Get kernel release version (alias for kernel attribute)."""
        return self.kernel
    
    def __str__(self):
        return f"{self.os_name} {self.os_version} (Kernel: {self.kernel})"


class PlatformDetector:
    """Platform detector for OS, kernel, BIOS, and network information."""
    
    @staticmethod
    def get_sbios_version() -> str:
        """Get SBIOS/BIOS version using dmidecode (Linux) or WMI (Windows).
        
        Returns:
            str: SBIOS version or "Unknown" if detection fails
            
        Note:
            Linux requires root/sudo for dmidecode; falls back to sysfs.
            Windows uses WMIC or PowerShell WMI queries.
        """
        import subprocess
        import platform
        
        os_type = platform.system()
        
        # Windows detection
        if os_type == "Windows":
            try:
                # Method 1: WMIC (Windows Management Instrumentation Command-line)
                result = subprocess.run(
                    ['wmic', 'bios', 'get', 'smbiosbiosversion'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    shell=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    # Skip header line and get version
                    if len(lines) > 1:
                        bios_version = lines[1].strip()
                        if bios_version:
                            return bios_version
            except Exception:
                pass
            
            try:
                # Method 2: PowerShell WMI query
                result = subprocess.run(
                    ['powershell', '-Command', 'Get-WmiObject', 'Win32_BIOS', '|', 'Select-Object', '-ExpandProperty', 'SMBIOSBIOSVersion'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except Exception:
                pass
        
        # Linux detection
        elif os_type == "Linux":
            import logging
            logger = logging.getLogger(__name__)
            
            # Method 1: dmidecode (requires root/sudo)
            try:
                logger.debug("Trying dmidecode for SBIOS...")
                result = subprocess.run(
                    ['dmidecode', '-s', 'bios-version'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    stderr=subprocess.PIPE
                )
                if result.returncode == 0 and result.stdout.strip():
                    logger.debug(f"SBIOS from dmidecode: {result.stdout.strip()}")
                    return result.stdout.strip()
                else:
                    logger.debug(f"dmidecode failed: returncode={result.returncode}, stderr={result.stderr[:100]}")
            except FileNotFoundError:
                logger.debug("dmidecode command not found")
            except (subprocess.TimeoutExpired, PermissionError) as e:
                logger.debug(f"dmidecode error: {e}")
            
            # Method 2: /sys/class/dmi/id/bios_version
            try:
                logger.debug("Trying /sys/class/dmi/id/bios_version...")
                with open('/sys/class/dmi/id/bios_version', 'r') as f:
                    bios_content = f.read().strip()
                    if bios_content:
                        logger.debug(f"SBIOS from sysfs: {bios_content}")
                        return bios_content
            except FileNotFoundError:
                logger.debug("/sys/class/dmi/id/bios_version not found")
            except PermissionError:
                logger.debug("/sys/class/dmi/id/bios_version permission denied")
            except IOError as e:
                logger.debug(f"/sys/class/dmi/id/bios_version error: {e}")
            
            # Method 3: /sys/devices/virtual/dmi/id/bios_version
            try:
                logger.debug("Trying /sys/devices/virtual/dmi/id/bios_version...")
                with open('/sys/devices/virtual/dmi/id/bios_version', 'r') as f:
                    bios_content = f.read().strip()
                    if bios_content:
                        logger.debug(f"SBIOS from alternate sysfs: {bios_content}")
                        return bios_content
            except FileNotFoundError:
                logger.debug("/sys/devices/virtual/dmi/id/bios_version not found")
            except PermissionError:
                logger.debug("/sys/devices/virtual/dmi/id/bios_version permission denied")
            except IOError as e:
                logger.debug(f"/sys/devices/virtual/dmi/id/bios_version error: {e}")
            
            logger.debug("All SBIOS detection methods failed")
        
        # macOS detection
        elif os_type == "Darwin":
            try:
                result = subprocess.run(
                    ['system_profiler', 'SPHardwareDataType'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # Parse for Boot ROM Version
                    import re
                    match = re.search(r'Boot ROM Version:\s*(.+)', result.stdout)
                    if match:
                        return match.group(1).strip()
            except Exception:
                pass
        
        return "Unknown"
    
    @staticmethod
    def get_system_ip() -> str:
        """Get system's primary IP address using socket routing.
        
        Returns:
            str: IP address or '0.0.0.0' if detection fails
        """
        try:
            # Create a socket to determine the primary network interface
            # This doesn't actually connect, just determines routing
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0)
            try:
                # Connect to a public DNS server (doesn't send data)
                s.connect(('8.8.8.8', 80))
                ip_address = s.getsockname()[0]
            except Exception:
                ip_address = '0.0.0.0'
            finally:
                s.close()
            
            return ip_address
            
        except Exception:
            # Fallback: try to get IP from hostname
            try:
                hostname = socket.gethostname()
                ip_address = socket.gethostbyname(hostname)
                # Avoid localhost
                if ip_address.startswith('127.'):
                    return '0.0.0.0'
                return ip_address
            except Exception:
                return '0.0.0.0'
    
    @staticmethod
    def detect() -> PlatformInfo:
        """Detect complete platform information.
        
        Returns:
            PlatformInfo: Platform details including OS, kernel, hostname, and BIOS
        """
        # Initialize with defaults
        os_name = "Unknown"
        os_version = "Unknown"
        kernel = "Unknown"
        hostname = "Unknown"
        architecture = "Unknown"
        sbios = "Unknown"
        
        # Get OS name
        try:
            os_name = platform.system()
        except Exception:
            pass
        
        # Get OS version
        try:
            if os_name == "Linux":
                try:
                    with open('/etc/os-release', 'r') as f:
                        os_release = f.read()
                    
                    # Extract NAME and VERSION_ID
                    import re
                    name_match = re.search(r'NAME="?([^"\n]+)"?', os_release)
                    version_match = re.search(r'VERSION_ID="?([^"\n]+)"?', os_release)
                    
                    if name_match:
                        os_name = name_match.group(1).strip()
                    if version_match:
                        os_version = version_match.group(1).strip()
                    else:
                        os_version = platform.release()
                except Exception:
                    os_version = platform.release()
            elif os_name == "Windows":
                # Windows version detection
                try:
                    os_version = platform.release()  # e.g., "10", "11"
                    # Get more detailed version if available
                    win_ver = platform.win32_ver()
                    if win_ver[0]:
                        os_version = f"{win_ver[0]} {win_ver[1]}"  # e.g., "10 10.0.19041"
                except Exception:
                    os_version = platform.release()
            elif os_name == "Darwin":
                # macOS version detection
                try:
                    os_version = platform.mac_ver()[0]
                except Exception:
                    os_version = platform.release()
            else:
                os_version = platform.release()
        except Exception:
            pass
        
        # Get kernel
        try:
            kernel = platform.release()
        except Exception:
            pass
        
        # Get hostname
        try:
            hostname = socket.gethostname()
        except Exception:
            pass
        
        # Get architecture
        try:
            architecture = platform.machine()
        except Exception:
            pass
        
        # Get SBIOS version using helper method
        try:
            sbios = PlatformDetector.get_sbios_version()
        except Exception:
            pass
        
        return PlatformInfo(
            os_name=os_name,
            os_version=os_version,
            kernel=kernel,
            hostname=hostname,
            architecture=architecture,
            sbios=sbios
        )