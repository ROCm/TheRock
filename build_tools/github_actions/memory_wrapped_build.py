#!/usr/bin/env python3
"""Wrapper script for build commands with automatic memory monitoring.

This script wraps common build commands with memory monitoring to help
identify out-of-memory issues in CI builds.

Usage:
    # Instead of: cmake --build build --target therock-archives
    # Use: python build_tools/github_actions/memory_wrapped_build.py cmake --build build --target therock-archives
    
    # Or with explicit phase name:
    # python build_tools/github_actions/memory_wrapped_build.py --phase "Build therock-archives" -- cmake --build build --target therock-archives
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path to import memory_monitor
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from memory_monitor import run_command_with_monitoring
except ImportError:
    print("ERROR: Could not import memory_monitor module", file=sys.stderr)
    sys.exit(1)


def detect_phase_name(command: list) -> str:
    """Attempt to detect a descriptive phase name from the command."""
    if not command:
        return "Build Phase"
    
    cmd_str = " ".join(command)
    
    # Detect common build phases
    if "cmake" in command[0].lower():
        if "--build" in command:
            # Extract target if specified
            if "--target" in command:
                try:
                    target_idx = command.index("--target")
                    if target_idx + 1 < len(command):
                        return f"Build Target: {command[target_idx + 1]}"
                except ValueError:
                    pass
            return "CMake Build"
        elif any(arg.startswith("-D") for arg in command):
            return "CMake Configure"
        else:
            return "CMake"
    
    elif "ctest" in command[0].lower():
        return "CTest"
    
    elif "ninja" in command[0].lower():
        return "Ninja Build"
    
    elif "make" in command[0].lower():
        return "Make Build"
    
    elif "fetch_sources" in cmd_str:
        return "Fetch Sources"
    
    elif "pytest" in cmd_str:
        return "Pytest"
    
    else:
        # Use the first part of the command
        return f"Build Phase: {command[0]}"


def main():
    parser = argparse.ArgumentParser(
        description="Wrap build commands with memory monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--phase",
        type=str,
        help="Name of the build phase (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.getenv("MEMORY_MONITOR_INTERVAL", "10")),
        help="Monitoring interval in seconds (default: 10)"
    )
    
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.getenv("MEMORY_MONITOR_LOG_DIR", "build/memory-logs")),
        help="Directory to write memory logs (default: build/memory-logs)"
    )
    
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable memory monitoring (useful for debugging)"
    )
    
    parser.add_argument(
        "command",
        nargs="*",
        help="Command to execute with monitoring"
    )
    
    args, unknown = parser.parse_known_args()
    
    # Combine parsed command with any unknown args (which are part of the command)
    command = args.command + unknown
    
    # Handle the -- separator if present
    if command and command[0] == "--":
        command = command[1:]
    
    if not command:
        parser.print_help()
        print("\nERROR: No command specified", file=sys.stderr)
        return 1
    
    # If monitoring is disabled, just run the command directly
    if args.no_monitor:
        print(f"[!] Memory monitoring disabled, executing command directly")
        result = subprocess.run(command)
        return result.returncode
    
    # Detect phase name if not specified
    phase_name = args.phase or detect_phase_name(command)
    
    # Create log directory and file
    log_dir = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize phase name for filename
    safe_phase_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in phase_name)
    safe_phase_name = safe_phase_name.replace(' ', '_')
    
    log_file = log_dir / f"{safe_phase_name}.jsonl"
    
    print(f"[*] Memory monitoring enabled: {phase_name}")
    print(f"[LOG] Logging to: {log_file}")
    
    # Run with monitoring
    return_code = run_command_with_monitoring(
        command=command,
        phase_name=phase_name,
        interval=args.interval,
        log_file=log_file,
    )
    
    return return_code


if __name__ == "__main__":
    sys.exit(main())

