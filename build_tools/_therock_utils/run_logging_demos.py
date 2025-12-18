#!/usr/bin/env python3
"""
Logging Framework Demo Runner

Runs sample applications that demonstrate TheRock's unified logging framework:
1. Package Installer - shows installation workflow with logging
2. Build System - shows build process with logging
3. GTest Demo - shows GTest execution with logging
4. CTest Demo - shows CTest execution with logging

All samples use the same logging framework with:
- Consistent formatting
- Structured data (extra fields)
- Performance timing
- Exception handling
"""


def main():
    """Run the logging framework demo with all sample applications"""
    print("\n" + "="*70)
    print("  UNIFIED LOGGING FRAMEWORK DEMO")
    print("  Showcasing same logging framework across different components")
    print("="*70 + "\n")
    
    print("All samples use TheRockLogger with consistent formatting,")
    print("structured data, performance timing, and error handling.\n")
    
    # Run sample 1: Package Installer
    print("\n" + "-"*70)
    print("  DEMO 1: Package Installer")
    print("-"*70)
    from sample_package_installer import main as installer_main
    installer_main()
    
    # Run sample 2: Build System
    print("\n" + "-"*70)
    print("  DEMO 2: Build System")
    print("-"*70)
    from sample_build_system import main as build_main
    build_main()
    
    # Run sample 3: GTest Demo
    print("\n" + "-"*70)
    print("  DEMO 3: GTest Framework")
    print("-"*70)
    from sample_gtest_demo import main as gtest_main
    gtest_main()
    
    # Run sample 4: CTest Demo
    print("\n" + "-"*70)
    print("  DEMO 4: CTest Framework")
    print("-"*70)
    from sample_ctest_demo import main as ctest_main
    ctest_main()
    
    print("\n" + "="*70)
    print("  KEY OBSERVATIONS:")
    print("="*70)
    print("✅ All samples use the same TheRockLogger")
    print("✅ Consistent log format across different components")
    print("✅ Structured data (extra fields) for better tracking")
    print("✅ Automatic timing with timed_operation context manager")
    print("✅ Test result parsing for GTest and CTest")
    print("✅ Exception handling with full tracebacks")
    print("\n📋 Configuration reference: logging_demo.yaml")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
