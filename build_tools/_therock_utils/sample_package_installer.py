#!/usr/bin/env python3
"""
Sample Package Installer - Demonstrates logging framework usage
"""

import time
import logging
from logging_config import get_logger, configure_root_logger


def download_packages(packages):
    """
    Sample package download with timed operations
    
    Demonstrates:
    - timed_operation: Automatic timing for download operations
    - Structured logging with detailed metadata (size, url, checksum)
    """
    logger = get_logger("sample.package_installer", component="PackageInstaller", operation="download")
    
    logger.info("Starting package download phase", extra={
        "total_packages": len(packages),
        "operation": "download",
        "download_source": "artifactory.amd.com"
    })
    
    for i, package in enumerate(packages, 1):
        # Simulate package metadata
        package_size_mb = 50 + (i * 25)
        package_url = f"https://repo.radeon.com/rocm/apt/{package}.deb"
        
        with logger.timed_operation(f"Downloading {package}"):
            logger.info(f"Downloading package {i}/{len(packages)}: {package}", extra={
                "package_name": package,
                "package_size_mb": package_size_mb,
                "download_url": package_url,
                "progress_pct": (i / len(packages)) * 100
            })
            
            # Simulate download
            time.sleep(0.4)
            
            logger.info(f"Package {package} downloaded successfully", extra={
                "package_name": package,
                "package_size_mb": package_size_mb,
                "checksum": f"sha256:{hash(package):016x}",
                "status": "downloaded"
            })
    
    logger.info("All packages downloaded successfully")
    
    # Display structured download metrics
    download_metrics = {
        "total_packages": len(packages),
        "total_size_mb": sum(50 + (i * 25) for i in range(1, len(packages) + 1)),
        "operation": "download_complete"
    }
    logger.log_dict(download_metrics, message="📊 Download Metrics:")


def install_packages(packages):
    """
    Sample package installation with logging
    
    Demonstrates:
    - timed_operation: Automatic timing for each package installation
    - Structured logging with extra fields including dependencies
    """
    logger = get_logger("sample.package_installer", component="PackageInstaller", operation="install")
    
    logger.info("Starting package installation", extra={
        "total_packages": len(packages),
        "operation": "install",
        "install_prefix": "/opt/rocm"
    })
    
    for i, package in enumerate(packages, 1):
        # Simulate package dependencies
        dependencies = ["libc6", "libstdc++6"] if i == 1 else [packages[0]]
        
        # timed_operation automatically logs:
        # - DEBUG: "Starting operation: Installing {package}"
        # - INFO: "Completed operation: Installing {package}" with extra={"duration_ms": X}
        with logger.timed_operation(f"Installing {package}"):
            logger.info(f"Installing package {i}/{len(packages)}: {package}", extra={
                "package_name": package,
                "progress": f"{i}/{len(packages)}",
                "dependencies": dependencies,
                "install_type": "fresh_install"
            })
            
            # Simulate dependency resolution
            time.sleep(0.2)
            logger.debug(f"Resolved {len(dependencies)} dependencies", extra={
                "package_name": package,
                "dependencies_count": len(dependencies)
            })
            
            # Simulate installation
            time.sleep(0.5)
            
            logger.info(f"Package {package} installed successfully", extra={
                "package_name": package,
                "status": "success",
                "installed_files_count": 150 + (i * 50),
                "install_path": f"/opt/rocm/lib/{package}"
            })
    
    logger.info("All packages installed successfully")
    
    # Display structured installation metrics
    install_metrics = {
        "total_packages": len(packages),
        "operation": "install_complete",
        "total_files_installed": sum(150 + (i * 50) for i in range(1, len(packages) + 1)),
        "install_prefix": "/opt/rocm"
    }
    logger.log_dict(install_metrics, message="📊 Installation Metrics:")


def verify_installation(packages):
    """
    Sample verification with error handling
    
    Demonstrates:
    - timed_operation: Automatic timing for verification
    - log_exception: Unified exception handling with traceback
    - Structured data: version info, file counts, checksums
    """
    logger = get_logger("sample.package_installer", component="PackageInstaller", operation="verify")
    
    logger.info("Verifying package installation", extra={
        "total_packages": len(packages),
        "verification_type": "post_install"
    })
    
    passed_count = 0
    failed_count = 0
    
    for i, package in enumerate(packages, 1):
        # Using timed_operation for automatic timing
        with logger.timed_operation(f"Verify {package}"):
            try:
                # Simulate verification steps
                logger.debug(f"Checking package files for {package}", extra={
                    "package_name": package,
                    "check_type": "file_integrity"
                })
                time.sleep(0.2)
                
                # Simulate version check
                version = f"6.{i}.0-{100+i}"
                logger.debug(f"Version check: {version}", extra={
                    "package_name": package,
                    "version": version
                })
                time.sleep(0.1)
                
                if "rocm" in package.lower():
                    passed_count += 1
                    logger.info(f"✅ {package} verification passed", extra={
                        "package_name": package,
                        "verification": "passed",
                        "version": version,
                        "files_verified": 150 + (i * 50),
                        "checksum_valid": True,
                        "install_path": f"/opt/rocm/lib/{package}"
                    })
                else:
                    raise ValueError(f"Package {package} not found in system (expected in /opt/rocm)")
            except Exception as e:
                failed_count += 1
                # Using log_exception for unified error handling
                logger.log_exception(e, f"❌ Verification failed for {package}", extra={
                    "package_name": package,
                    "verification": "failed",
                    "error_type": type(e).__name__,
                    "expected_path": f"/opt/rocm/lib/{package}"
                })
    
    # Summary log with structured data
    logger.info("Verification phase completed")
    
    # Display structured verification metrics
    verification_metrics = {
        "total_packages": len(packages),
        "passed": passed_count,
        "failed": failed_count,
        "success_rate_pct": round((passed_count / len(packages)) * 100, 2) if packages else 0
    }
    logger.log_dict(verification_metrics, message="📊 Verification Metrics:")


def main():
    """Main demo function"""
    # Explicitly set DEBUG level to see timed_operation's detailed logs
    # Note: Logging auto-configures to INFO by default, but we want DEBUG for this demo
    configure_root_logger(level=logging.DEBUG)
    
    print("\n" + "="*60)
    print("  Sample 1: Package Installer with Logging")
    print("  Demonstrates: Timed Operations, Structured Data, Exception Handling")
    print("="*60 + "\n")
    
    # Setup logger
    logger = get_logger("sample.package_installer", component="PackageInstaller")
    logger.info("Package Installer Demo Started", extra={
        "demo_version": "2.0",
        "features": ["timed_operations", "structured_logging", "exception_handling"]
    })
    logger.debug("🔧 Logging initialized at DEBUG level - you'll see automatic timing logs")
    
    # Demo packages
    packages = [
        "rocm-core",
        "rocm-hip-runtime",
        "pytorch-rocm"
    ]
    
    logger.info("Starting installation workflow", extra={
        "packages": packages,
        "phases": ["download", "install", "verify"]
    })
    
    # Run complete installation workflow with nested timed operations
    with logger.timed_operation("Complete Installation Workflow"):
        download_packages(packages)
        install_packages(packages)
        verify_installation(packages)
    
    logger.info("Package Installer Demo Completed", extra={
        "total_packages": len(packages),
        "all_phases_completed": True
    })
    print("\n" + "="*60)
    print("  ✅ Demo completed - Check logs above for:")
    print("     - Timed operation durations (duration_ms)")
    print("     - Structured data fields (package metadata)")
    print("     - Exception logs with full tracebacks")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

