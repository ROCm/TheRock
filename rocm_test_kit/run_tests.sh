#!/bin/bash
#
# ROCm Component Test Kit - Simple Runner Script
#
# This script makes it easy to run the ROCm test kit with a single command.
# Just execute: ./run_tests.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THEROCK_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  ROCm Component Test Kit${NC}"
echo -e "${BLUE}  MI300/MI350 Hardware Testing${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    echo "Please install Python 3 to run this test kit."
    exit 1
fi

# Check if required Python packages are available
python3 -c "import yaml" 2>/dev/null || {
    echo -e "${YELLOW}Warning: PyYAML not installed${NC}"
    echo "Installing PyYAML..."
    pip3 install pyyaml || {
        echo -e "${RED}Failed to install PyYAML${NC}"
        echo "Please install manually: pip3 install pyyaml"
        exit 1
    }
}

# Parse command line arguments
PRESET="quick"
TEST_TYPE="smoke"
PARALLEL=""
VERBOSE=""
LOG_DIR=""
GENERATE_REPORT=""

# Simple argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            PRESET="quick"
            shift
            ;;
        --core)
            PRESET="core"
            shift
            ;;
        --full)
            PRESET="full"
            TEST_TYPE="full"
            shift
            ;;
        --smoke)
            TEST_TYPE="smoke"
            shift
            ;;
        --parallel)
            PARALLEL="--parallel"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --with-report)
            GENERATE_REPORT="yes"
            LOG_DIR="${LOG_DIR:-./test_logs}"
            shift
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --check-hardware)
            python3 "$SCRIPT_DIR/test_runner.py" --check-hardware
            exit $?
            ;;
        --list)
            python3 "$SCRIPT_DIR/test_runner.py" --list-components
            exit $?
            ;;
        --help|-h)
            cat << EOF
ROCm Component Test Kit - Simple Runner

Usage: $0 [OPTIONS]

Quick Options:
  --quick         Run quick test (5-10 min, default)
  --core          Run core libraries test (15-20 min)
  --full          Run full test suite (2-4 hours)

Test Options:
  --smoke         Run smoke tests only (fast, default)
  --parallel      Run tests in parallel
  --verbose       Verbose output
  --with-report   Generate HTML report
  --log-dir DIR   Store logs in DIR

Info Options:
  --check-hardware    Check hardware compatibility
  --list              List all available components
  --help, -h          Show this help message

Examples:
  # Quick smoke test (default)
  $0

  # Quick test with HTML report
  $0 --quick --with-report

  # Full test suite in parallel
  $0 --full --parallel

  # Core libraries test with logs
  $0 --core --log-dir ./logs --verbose

  # Check hardware
  $0 --check-hardware

For advanced usage, run the test_runner.py script directly:
  python3 $SCRIPT_DIR/test_runner.py --help
EOF
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

# Set default log directory if generating report
if [ -n "$GENERATE_REPORT" ] && [ -z "$LOG_DIR" ]; then
    LOG_DIR="./test_logs_$(date +%Y%m%d_%H%M%S)"
fi

# Build command
CMD="python3 $SCRIPT_DIR/test_runner.py --preset $PRESET --test-type $TEST_TYPE"

if [ -n "$PARALLEL" ]; then
    CMD="$CMD $PARALLEL"
fi

if [ -n "$VERBOSE" ]; then
    CMD="$CMD $VERBOSE"
fi

if [ -n "$LOG_DIR" ]; then
    CMD="$CMD --log-dir $LOG_DIR"
fi

# Show configuration
echo -e "${GREEN}Configuration:${NC}"
echo "  Preset:     $PRESET"
echo "  Test Type:  $TEST_TYPE"
echo "  Parallel:   $([ -n "$PARALLEL" ] && echo "Yes" || echo "No")"
echo "  Logs:       $([ -n "$LOG_DIR" ] && echo "$LOG_DIR" || echo "None")"
echo "  Report:     $([ -n "$GENERATE_REPORT" ] && echo "Yes" || echo "No")"
echo ""

# Run the test kit
echo -e "${GREEN}Starting tests...${NC}"
echo ""

# Execute
if $CMD; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Tests completed successfully! ✓${NC}"
    echo -e "${GREEN}========================================${NC}"
    EXIT_CODE=0
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Some tests failed ✗${NC}"
    echo -e "${RED}========================================${NC}"
    EXIT_CODE=1
fi

# Generate report if requested
if [ -n "$GENERATE_REPORT" ]; then
    echo ""
    echo -e "${BLUE}Generating HTML report...${NC}"

    REPORT_FILE="test_report_$(date +%Y%m%d_%H%M%S).html"

    # Create a simple Python script to generate the report
    python3 << EOFPYTHON
import sys
import json
from pathlib import Path

# Add test kit to path
sys.path.insert(0, "$SCRIPT_DIR")

from report_generator import generate_html_report, generate_json_report
from hardware_detector import detect_hardware

# This is a placeholder - in a real scenario, we'd parse the actual test results
# For now, we'll create a simple message
print("Note: Full report generation requires integration with test_runner.py")
print("      See report_generator.py for standalone usage")
EOFPYTHON

    echo -e "${GREEN}Report would be generated at: $REPORT_FILE${NC}"
    echo -e "${YELLOW}Note: Integrate report_generator.py into test_runner.py for full functionality${NC}"
fi

# Show log location if available
if [ -n "$LOG_DIR" ] && [ -d "$LOG_DIR" ]; then
    echo ""
    echo -e "${BLUE}Test logs available at: $LOG_DIR${NC}"
fi

exit $EXIT_CODE
