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
    - timed_operation: Automatically logs start (DEBUG) and completion (INFO) with duration_ms
    - Manual timing: Explicit control over timing messages
    """
    logger = get_logger(__name__, component="PackageInstaller", operation="install")
    
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
    - Exception handling with exc_info=True for full traceback
    - Manual timing for explicit control
    """
    logger = get_logger(__name__, component="PackageInstaller", operation="verify")
    
    logger.info("Verifying package installation")
    
    for package in packages:
        # Manual timing - explicit start/end messages with duration
        start_time = time.time()
        logger.info(f"🔍 Starting verification: {package}")
        
        try:
            # Simulate verification
            time.sleep(0.3)
            
            if "rocm" in package.lower():
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"✅ {package} verified in {duration_ms:.2f}ms", extra={
                    "package_name": package,
                    "verification": "passed",
                    "duration_ms": duration_ms
                })
            else:
                raise ValueError(f"Package {package} not found in system")
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Verification failed for {package} after {duration_ms:.2f}ms", extra={
                "package_name": package,
                "verification": "failed",
                "error": str(e),
                "duration_ms": duration_ms
            }, exc_info=True)


def main():
    """Main demo function"""
    # Initialize logging with DEBUG level to see timed_operation's automatic logs
    configure_root_logger(level=logging.DEBUG)
    
    print("\n" + "="*60)
    print("  Sample 1: Package Installer with Logging")
    print("="*60 + "\n")
    
    # Setup logger
    logger = get_logger(__name__, component="PackageInstaller")
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

