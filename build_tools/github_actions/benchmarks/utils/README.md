# Utils Module

Utility modules organized into logical subdirectories for maintainability and scalability.

## Structure

```
benchmarks/utils/
├── __init__.py              # Public exports
├── benchmark_client.py      # Main BenchmarkClient API
├── constants.py             # Framework constants
├── exceptions.py            # Custom exceptions
├── logger.py                # Logging configuration
├── README.md                # This file
│
├── config/                  # Configuration management
│   ├── __init__.py
│   ├── config_helper.py     # Config utilities
│   ├── config_parser.py     # YAML parsing with env vars
│   └── config_validator.py  # Schema validation
│
├── results/                 # Results handling
│   ├── __init__.py
│   ├── results_handler.py   # Formatting and saving
│   ├── results_api.py       # API submission
│   └── schemas/
│       └── payload_schema.json  # API payload validation schema
│
└── system/                  # System detection
    ├── __init__.py
    ├── hardware.py          # CPU/GPU detection
    ├── platform.py          # OS/platform detection
    ├── rocm_detector.py     # ROCm detection
    └── system_detector.py   # Main orchestrator
```

## Usage

### From Benchmark Scripts

Benchmark scripts add `benchmarks/` to `sys.path`, then import:

```python
# Import path setup (already done in benchmark_base.py)
sys.path.insert(0, str(Path(__file__).parent.parent))  # Adds benchmarks/ to path

# Core utilities
from utils.logger import log
from utils.constants import Constants
from utils.exceptions import ConfigurationError

# Main API classes
from utils.benchmark_client import BenchmarkClient
from utils.system.system_detector import SystemDetector
from utils.config.config_helper import ConfigHelper
from utils.results.results_handler import ResultsHandler
```

### Subdirectory Imports

```python
# Configuration
from utils.config import ConfigHelper, ConfigParser, ConfigValidator

# System detection
from utils.system import (
    SystemDetector,
    HardwareDetector,
    ROCmDetector,
    PlatformDetector,
)

# Results handling
from utils.results import ResultsHandler, ResultsAPI
```

## Modules

### Root Level

- **constants.py** - Framework constants and defaults
- **exceptions.py** - Custom exception classes
- **logger.py** - Logging configuration
- **benchmark_client.py** - Main BenchmarkClient API

### Config

Configuration loading, parsing, and validation.

- **config_helper.py** - High-level config utilities
- **config_parser.py** - YAML parser with environment variable expansion
- **config_validator.py** - JSON Schema validation

### System

Platform, hardware, and ROCm detection.

- **system_detector.py** - Main orchestrator
- **hardware.py** - CPU and GPU detection
- **platform.py** - OS, kernel, SBIOS detection
- **rocm_detector.py** - ROCm version and build info

### Results

Test results formatting, saving, and API submission.

- **results_handler.py** - Results formatting and local saving
- **results_api.py** - REST API client

## Adding New Modules

### To Existing Subdirectory

1. Create file in appropriate subdirectory
1. Add exports to subdirectory's `__init__.py`
1. Optionally add to `utils/__init__.py` for backward compatibility

### New Subdirectory

1. Create directory with `__init__.py`
1. Add modules
1. Update `utils/__init__.py` for commonly used classes

## Testing

```bash
# Run from project root
cd /path/to/TheRock

# Run a benchmark test (imports are handled internally)
python build_tools/github_actions/benchmarks/scripts/test_rocfft_benchmark.py

# Verify utils imports work
cd build_tools/github_actions/benchmarks
python -c "from utils.logger import log; print('✓ Utils imports working')"
```
