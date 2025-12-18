#!/usr/bin/env python3
"""
Sample Package Installer - Demonstrates logging framework usage
"""

import time
import logging
from logging_config import get_logger, configure_root_logger


def install_packages(packages):
    """
    Sample package installation with logging
    
    Demonstrates:
    - timed_operation: Automatic timing for each package installation
    - Structured logging with extra fields
    """
    logger = get_logger("sample.package_installer", component="PackageInstaller", operation="install")
    
    logger.info("Starting package installation", extra={
        "total_packages": len(packages),
        "operation": "install"
    })
    
    for i, package in enumerate(packages, 1):
        # timed_operation automatically logs:
        # - DEBUG: "Starting operation: Installing {package}"
        # - INFO: "Completed operation: Installing {package}" with extra={"duration_ms": X}
        with logger.timed_operation(f"Installing {package}"):
            logger.info(f"Installing package {i}/{len(packages)}: {package}", extra={
                "package_name": package,
                "progress": f"{i}/{len(packages)}"
            })
            
            # Simulate installation
            time.sleep(0.5)
            
            logger.info(f"Package {package} installed successfully", extra={
                "package_name": package,
                "status": "success"
            })
    
    logger.info("All packages installed successfully", extra={
        "total_packages": len(packages),
        "operation": "install_complete"
    })


def verify_installation(packages):
    """
    Sample verification with error handling
    
    Demonstrates:
    - timed_operation: Automatic timing for verification
    - log_exception: Unified exception handling with traceback
    """
    logger = get_logger("sample.package_installer", component="PackageInstaller", operation="verify")
    
    logger.info("Verifying package installation")
    
    for package in packages:
        # Using timed_operation for automatic timing
        with logger.timed_operation(f"Verify {package}"):
            try:
                # Simulate verification
                time.sleep(0.3)
                
                if "rocm" in package.lower():
                    logger.info(f"✅ {package} verification passed", extra={
                        "package_name": package,
                        "verification": "passed"
                    })
                else:
                    raise ValueError(f"Package {package} not found in system")
            except Exception as e:
                # Using log_exception for unified error handling
                logger.log_exception(e, f"❌ Verification failed for {package}", extra={
                    "package_name": package,
                    "verification": "failed"
                })


def main():
    """Main demo function"""
    # Explicitly set DEBUG level to see timed_operation's detailed logs
    # Note: Logging auto-configures to INFO by default, but we want DEBUG for this demo
    configure_root_logger(level=logging.DEBUG)
    
    print("\n" + "="*60)
    print("  Sample 1: Package Installer with Logging")
    print("="*60 + "\n")
    
    # Setup logger
    logger = get_logger("sample.package_installer", component="PackageInstaller")
    logger.info("Package Installer Demo Started")
    logger.debug("🔧 Logging initialized at DEBUG level - you'll see automatic timing logs")
    
    # Demo packages
    packages = [
        "rocm-core",
        "rocm-hip-runtime",
        "pytorch-rocm"
    ]
    
    # Run installation
    with logger.timed_operation("Complete Installation"):
        install_packages(packages)
        verify_installation(packages)
    
    logger.info("Package Installer Demo Completed")
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()

