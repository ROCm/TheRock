#!/usr/bin/env python3
"""
Sample Package Installer - Demonstrates logging framework usage
"""

import time
from logging_config import get_logger


def install_packages(packages):
    """Sample package installation with logging"""
    logger = get_logger(__name__, component="PackageInstaller", operation="install")
    
    logger.info("Starting package installation", extra={
        "total_packages": len(packages),
        "operation": "install"
    })
    
    for i, package in enumerate(packages, 1):
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
    """Sample verification with error handling"""
    logger = get_logger(__name__, component="PackageInstaller", operation="verify")
    
    logger.info("Verifying package installation")
    
    for package in packages:
        try:
            # Simulate verification
            if "rocm" in package.lower():
                logger.info(f"✓ {package} verified successfully", extra={
                    "package_name": package,
                    "verification": "passed"
                })
            else:
                raise ValueError(f"Package {package} not found")
        except Exception as e:
            logger.error(f"Verification failed for {package}", extra={
                "package_name": package,
                "verification": "failed",
                "error": str(e)
            }, exc_info=True)


def main():
    """Main demo function"""
    print("\n" + "="*60)
    print("  Sample 1: Package Installer with Logging")
    print("="*60 + "\n")
    
    # Setup logger
    logger = get_logger(__name__, component="PackageInstaller")
    logger.info("Package Installer Demo Started")
    
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

