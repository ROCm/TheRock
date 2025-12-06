"""ROCm detection for version, build info, and compatibility checking."""

import subprocess
import re
import os
from typing import Dict, Optional, Tuple
import logging
from packaging import version

from ..exceptions import ROCmVersionError


def _get_rocm_tool_path(tool_name: str) -> str:
    """Get full path to ROCm tool using THEROCK_BIN_DIR.
    
    Args:
        tool_name: Name of the tool ('rocm-smi', 'amd-smi', or 'rocminfo')
    
    Returns:
        str: Full path to tool if THEROCK_BIN_DIR is set, otherwise just tool name
    """
    therock_bin_dir = os.getenv('THEROCK_BIN_DIR')
    if therock_bin_dir:
        return os.path.join(therock_bin_dir, tool_name)
    return tool_name


def _get_rocm_path() -> str:
    """Get ROCm base path using THEROCK_BIN_DIR or fallback to /opt/rocm.
    
    Returns:
        str: ROCm base path (e.g., /opt/rocm-7.1.0 or /opt/rocm)
    """
    therock_bin_dir = os.getenv('THEROCK_BIN_DIR')
    if therock_bin_dir:
        # THEROCK_BIN_DIR is typically /opt/rocm-x.y.z/bin, so parent is /opt/rocm-x.y.z
        return os.path.dirname(therock_bin_dir)
    return '/opt/rocm'


class ROCmDetector:
    """ROCm detector for version, build type, package manager, and compatibility validation."""
    
    @staticmethod
    def detect_rocm_info() -> Dict[str, str]:
        """Detect ROCm build information.
        
        Returns:
            Dictionary with ROCm build information
        """
        logger = logging.getLogger(__name__)
        
        build_info = {
            "rocm_version": "Unknown",
            "rocm_build_type": "Unknown",
            "rocm_build_lib_type": "Unknown",
            "rocm_package_manager": "Unknown",
            "rocm_package_manager_version": "Unknown",
            "install_type": "Unknown"
        }
        
        # Detect ROCm version
        rocm_version = ROCmDetector._detect_rocm_version()
        if rocm_version:
            build_info["rocm_version"] = rocm_version
            logger.debug(f"ROCm version: {rocm_version}")
        
        # Detect package manager
        pkg_manager, pkg_version = ROCmDetector._detect_package_manager()
        if pkg_manager:
            build_info["rocm_package_manager"] = pkg_manager
            build_info["rocm_package_manager_version"] = pkg_version
            logger.debug(f"Package manager: {pkg_manager} {pkg_version}")
        
        # Detect install type
        install_type = ROCmDetector._detect_install_type()
        if install_type:
            build_info["install_type"] = install_type
            logger.debug(f"Install type: {install_type}")
        
        # Detect build type
        build_type = ROCmDetector._detect_build_type()
        if build_type:
            build_info["rocm_build_type"] = build_type
            logger.debug(f"Build type: {build_type}")
        
        # Detect library type
        lib_type = ROCmDetector._detect_lib_type()
        if lib_type:
            build_info["rocm_build_lib_type"] = lib_type
            logger.debug(f"Library type: {lib_type}")
        
        return build_info
    
    @staticmethod
    def _detect_rocm_version() -> Optional[str]:
        """Detect ROCm version using multiple methods.
        
        Returns:
            str: Version in format X.Y.Z-BUILD (e.g., "7.10.1-36") or None
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Method 1: Try $ROCM_PATH/.info/version-rocm (most complete - includes build number)
            rocm_path = _get_rocm_path()
            version_rocm_file = os.path.join(rocm_path, '.info', 'version-rocm')
            logger.debug(f"Trying {version_rocm_file} file...")
            if os.path.exists(version_rocm_file):
                with open(version_rocm_file, 'r') as f:
                    rocm_ver = f.read().strip()
                    if rocm_ver:
                        logger.debug(f"ROCm version from version-rocm file: {rocm_ver}")
                        return rocm_ver
            else:
                logger.debug(f"{version_rocm_file} not found")
        except Exception as e:
            logger.debug(f"Error reading version-rocm file: {e}")
        
        try:
            # Method 2: Try $ROCM_PATH/.info/version file
            rocm_path = _get_rocm_path()
            version_file = os.path.join(rocm_path, '.info', 'version')
            logger.debug(f"Trying {version_file} file...")
            if os.path.exists(version_file):
                with open(version_file, 'r') as f:
                    rocm_ver = f.read().strip()
                    if rocm_ver:
                        logger.debug(f"ROCm version from version file: {rocm_ver}")
                        # Check if it has build number, if not try to append from version-rocm
                        if '-' not in rocm_ver:
                            logger.debug(f"Version {rocm_ver} missing build number, trying to get full version...")
                            # Try to get build number from version-rocm
                            version_rocm_file = os.path.join(rocm_path, '.info', 'version-rocm')
                            if os.path.exists(version_rocm_file):
                                try:
                                    with open(version_rocm_file, 'r') as f2:
                                        full_ver = f2.read().strip()
                                        if full_ver and '-' in full_ver:
                                            logger.debug(f"Using full version from version-rocm: {full_ver}")
                                            return full_ver
                                except Exception:
                                    pass
                        return rocm_ver
            else:
                logger.debug(f"{version_file} not found")
        except Exception as e:
            logger.debug(f"Error reading version file: {e}")
        
        try:
            # Method 3: Try amd-smi version
            amd_smi_cmd = _get_rocm_tool_path('amd-smi')
            logger.debug(f"Trying {amd_smi_cmd} version for ROCm detection...")
            result = subprocess.run(
                [amd_smi_cmd, 'version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.debug(f"amd-smi version output:\n{result.stdout[:500]}")
                # Look for "ROCm version:" field
                match = re.search(r'ROCm version:\s*(\S+)', result.stdout, re.IGNORECASE)
                if match:
                    rocm_ver = match.group(1)
                    logger.debug(f"ROCm version from amd-smi: {rocm_ver}")
                    return rocm_ver
            else:
                logger.debug(f"amd-smi version failed with exit code {result.returncode}")
        except FileNotFoundError:
            logger.debug("amd-smi command not found")
        except Exception as e:
            logger.debug(f"amd-smi version error: {e}")
        
        try:
            # Method 4: Try rocminfo
            rocminfo_cmd = _get_rocm_tool_path('rocminfo')
            logger.debug(f"Trying {rocminfo_cmd} for ROCm detection...")
            result = subprocess.run(
                [rocminfo_cmd],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Look for "Runtime Version" or similar
                match = re.search(r'Runtime Version:\s*(\S+)', result.stdout)
                if match:
                    rocm_ver = match.group(1)
                    logger.debug(f"ROCm version from rocminfo: {rocm_ver}")
                    return rocm_ver
            else:
                logger.debug(f"rocminfo failed with exit code {result.returncode}")
        except FileNotFoundError:
            logger.debug("rocminfo command not found")
        except Exception as e:
            logger.debug(f"rocminfo error: {e}")
        
        try:
            # Method 5: Try rocm-smi --version
            rocm_smi_cmd = _get_rocm_tool_path('rocm-smi')
            logger.debug(f"Trying {rocm_smi_cmd} --version...")
            result = subprocess.run(
                [rocm_smi_cmd, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                match = re.search(r'version:\s*(\S+)', result.stdout, re.IGNORECASE)
                if match:
                    rocm_ver = match.group(1)
                    logger.debug(f"ROCm version from rocm-smi: {rocm_ver}")
                    return rocm_ver
            else:
                logger.debug(f"rocm-smi --version failed with exit code {result.returncode}")
        except FileNotFoundError:
            logger.debug("rocm-smi command not found")
        except Exception as e:
            logger.debug(f"rocm-smi error: {e}")
        
        logger.debug("All ROCm version detection methods failed")
        return None
    
    @staticmethod
    def _detect_package_manager() -> tuple:
        """Detect system package manager (dpkg or rpm) and version.
        
        Returns:
            tuple: (package_manager_name, version) or ("Unknown", "Unknown")
        """
        # Check for dpkg (Debian/Ubuntu)
        try:
            result = subprocess.run(
                ['dpkg', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                match = re.search(r'version\s+(\S+)', result.stdout)
                version = match.group(1) if match else "Unknown"
                return ("dpkg", version)
        except Exception:
            pass
        
        # Check for rpm (RHEL/CentOS/SLES)
        try:
            result = subprocess.run(
                ['rpm', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                match = re.search(r'version\s+(\S+)', result.stdout)
                version = match.group(1) if match else "Unknown"
                return ("rpm", version)
        except Exception:
            pass
        
        return ("Unknown", "Unknown")
    
    @staticmethod
    def _detect_install_type() -> Optional[str]:
        """Detect ROCm installation type (package manager or source build)."""
        # Check if installed via package manager
        try:
            # Check for dpkg package
            result = subprocess.run(
                ['dpkg', '-l', 'rocm-dev'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and 'ii' in result.stdout:
                return "package"
        except Exception:
            pass
        
        try:
            # Check for rpm package
            result = subprocess.run(
                ['rpm', '-q', 'rocm-dev'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "package"
        except Exception:
            pass
        
        # Check if ROCm path exists (could be source or package)
        rocm_path = _get_rocm_path()
        if os.path.exists(rocm_path):
            # If package manager doesn't show it, likely source
            return "source"
        
        return "Unknown"
    
    @staticmethod
    def _detect_build_type() -> Optional[str]:
        """Detect ROCm build type (Release, Debug, or TheRock)."""
        # Check for debug symbols or debug libraries
        try:
            rocm_path = _get_rocm_path()
            rocm_lib_path = os.path.join(rocm_path, 'lib')
            if os.path.exists(rocm_lib_path):
                # Check if debug libraries exist
                for file in os.listdir(rocm_lib_path):
                    if 'debug' in file.lower() or file.endswith('.debug'):
                        return "Debug"
                
                # Default to TheRock if no debug indicators
                return "TheRock"
        except Exception:
            pass
        
        return "Unknown"
    
    @staticmethod
    def _detect_lib_type() -> Optional[str]:
        """Detect ROCm library type (static, shared, or both)."""
        try:
            rocm_path = _get_rocm_path()
            rocm_lib_path = os.path.join(rocm_path, 'lib')
            if os.path.exists(rocm_lib_path):
                has_static = False
                has_shared = False
                
                for file in os.listdir(rocm_lib_path):
                    if file.endswith('.a'):
                        has_static = True
                    if file.endswith('.so') or '.so.' in file:
                        has_shared = True
                
                if has_static and has_shared:
                    return "both"
                elif has_shared:
                    return "shared"
                elif has_static:
                    return "static"
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def check_version_compatibility(required_version: str, detected_version: str) -> Tuple[bool, str]:
        """Check if detected ROCm version meets minimum requirement.
        
        Args:
            required_version: Minimum required version (e.g., "5.7.0")
            detected_version: Detected ROCm version
            
        Returns:
            tuple: (is_compatible: bool, message: str)
        """
        if detected_version == "Unknown":
            return False, "ROCm version could not be detected"
        
        if required_version == "Unknown" or not required_version:
            return True, f"ROCm {detected_version} detected (no minimum version specified)"
        
        try:
            required = version.parse(required_version)
            detected = version.parse(detected_version)
            
            if detected >= required:
                return True, f"ROCm {detected_version} meets requirement (>= {required_version})"
            else:
                return False, f"ROCm {detected_version} is too old (requires >= {required_version})"
                
        except version.InvalidVersion as e:
            return False, f"Invalid version format: {e}"
        except Exception as e:
            return False, f"Version comparison failed: {e}"
    
    @staticmethod
    def validate_rocm_version(required_version: str, detected_version: str) -> None:
        """Validate ROCm version compatibility and raise exception if incompatible.
        
        Args:
            required_version: Minimum required version
            detected_version: Detected ROCm version
            
        Raises:
            ROCmVersionError: If detected version is older than required
        """
        is_compatible, message = ROCmDetector.check_version_compatibility(
            required_version,
            detected_version
        )
        
        if not is_compatible:
            from ..logger import log
            log.error("✗ ROCm Version Incompatible")
            log.error(f"  Detected: ROCm {detected_version}")
            log.error(f"  Required: ROCm {required_version}+")
            log.error("  Solutions:")
            log.error(f"    1. Upgrade ROCm to {required_version} or higher")
            log.error("    2. Visit: https://rocm.docs.amd.com/")
            log.error("    3. Check release notes for upgrade instructions")
            log.error("    4. Adjust MinROCmVersion in config if appropriate")
            raise ROCmVersionError(message)
    
    @staticmethod
    def get_version_info(rocm_version: str) -> Dict[str, any]:
        """Parse ROCm version string into major, minor, and patch components.
        
        Args:
            rocm_version: ROCm version string (e.g., "6.0.2")
            
        Returns:
            Dict: Version components with keys: major, minor, patch, string
        """
        if rocm_version == "Unknown":
            return {
                'major': 0,
                'minor': 0,
                'patch': 0,
                'string': 'Unknown'
            }
        
        try:
            parsed = version.parse(rocm_version)
            # Handle both Version and LegacyVersion
            if hasattr(parsed, 'major'):
                return {
                    'major': parsed.major,
                    'minor': parsed.minor if hasattr(parsed, 'minor') else 0,
                    'patch': parsed.micro if hasattr(parsed, 'micro') else 0,
                    'string': rocm_version
                }
            else:
                # Fallback for legacy versions
                parts = rocm_version.split('.')
                return {
                    'major': int(parts[0]) if len(parts) > 0 else 0,
                    'minor': int(parts[1]) if len(parts) > 1 else 0,
                    'patch': int(parts[2]) if len(parts) > 2 else 0,
                    'string': rocm_version
                }
        except Exception:
            return {
                'major': 0,
                'minor': 0,
                'patch': 0,
                'string': rocm_version
            }