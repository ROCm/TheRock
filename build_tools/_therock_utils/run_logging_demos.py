#!/usr/bin/env python3
"""
Logging Framework Demo Runner

Runs sample applications that demonstrate TheRock's unified logging framework:
1. Package Installer - shows installation workflow with logging
2. Build System - shows build process with logging

All samples use the same logging framework with:
- Consistent formatting
- Structured data (extra fields)
- Performance timing
- Exception handling

Note: For GTest/CTest demos, see the component-specific mock tests in:
  build_tools/github_actions/test_executable_scripts/demo_test_*.py
"""


def main():
    """Run the logging framework demo with core sample applications"""
    print("\n" + "="*70)
    print("  UNIFIED LOGGING FRAMEWORK DEMO")
    print("  Core Samples: Package Installer + Build System")
    print("="*70 + "\n")
    
    print("All samples use TheRockLogger with consistent formatting,")
    print("structured data, performance timing, and error handling.\n")
    
    # Run sample 1: Package Installer
    print("\n" + "-"*70)
    print("  SAMPLE 1: Package Installer")
    print("-"*70)
    from sample_package_installer import main as installer_main
    installer_main()
    
    # Run sample 2: Build System
    print("\n" + "-"*70)
    print("  SAMPLE 2: Build System")
    print("-"*70)
    from sample_build_system import main as build_main
    build_main()
    
    print("\n" + "="*70)
    print("  KEY OBSERVATIONS:")
    print("="*70)
    print("✅ All samples use the same TheRockLogger")
    print("✅ Consistent log format across different components")
    print("✅ Structured data (extra fields) for better tracking")
    print("✅ Automatic timing with timed_operation context manager")
    print("✅ Exception handling with full tracebacks")
    print("\n📋 For GTest/CTest demos, see component-specific mocks:")
    print("   - demo_test_rocroller.py (GTest integration)")
    print("   - demo_test_rocwmma.py (CTest integration)")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
