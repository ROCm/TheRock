#!/usr/bin/env python3
"""
Logging Framework Demo Runner

Runs two sample applications that demonstrate TheRock's unified logging framework:
1. Package Installer - shows installation workflow with logging
2. Build System - shows build process with logging

Both samples use the same logging framework with:
- Consistent formatting
- Structured data (extra fields)
- Performance timing
- Exception handling
"""


def main():
    """Run the logging framework demo with both sample applications"""
    print("\n" + "="*70)
    print("  UNIFIED LOGGING FRAMEWORK DEMO")
    print("  Showcasing same logging framework across different components")
    print("="*70 + "\n")
    
    print("Both samples use TheRockLogger with consistent formatting,")
    print("structured data, performance timing, and error handling.\n")
    
    # Run sample 1: Package Installer
    from sample_package_installer import main as installer_main
    installer_main()
    
    # Run sample 2: Build System
    from sample_build_system import main as build_main
    build_main()
    
    print("\n" + "="*70)
    print("  KEY OBSERVATIONS:")
    print("="*70)
    print("✅ Both samples use the same TheRockLogger")
    print("✅ Consistent log format across different components")
    print("✅ Structured data (extra fields) for better tracking")
    print("✅ Automatic timing with timed_operation context manager")
    print("✅ Manual timing examples for explicit control")
    print("✅ Exception handling with full tracebacks")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
