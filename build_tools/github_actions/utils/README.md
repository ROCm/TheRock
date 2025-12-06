# Utils Module

Utility modules organized into logical subdirectories for maintainability and scalability.

## Structure

```
utils/
├── __init__.py              # Public exports
├── constants.py             # Framework constants
├── exceptions.py            # Custom exceptions
├── logger.py                # Logging configuration
├── test_client.py           # Main TestClient API
│
├── config/                  # Configuration management
│   ├── config_helper.py     # Config utilities
│   ├── config_parser.py     # YAML parsing with env vars
│   └── config_validator.py  # Schema validation
│
├── system/                  # System detection
│   ├── system_detector.py   # Main orchestrator
│   ├── hardware.py          # CPU/GPU detection
│   ├── platform.py          # OS/platform detection
│   └── rocm_detector.py     # ROCm detection
│
└── results/                 # Results handling
    ├── results_handler.py   # Formatting and saving
    └── results_api.py       # API submission
```

## Usage

### Standard Imports

```python
# Main API
from utils import TestClient, SystemDetector, ConfigHelper, ResultsHandler

# Core utilities
from utils.logger import log
from utils.constants import Constants
from utils.exceptions import ConfigurationError
```

### Subdirectory Imports

```python
# Configuration
from utils.config import ConfigHelper, ConfigParser, ConfigValidator

# System detection
from utils.system import SystemDetector, HardwareDetector, ROCmDetector

# Results handling
from utils.results import ResultsHandler, ResultsAPI
```

## Modules

### Root Level
- **constants.py** - Framework constants and defaults
- **exceptions.py** - Custom exception classes
- **logger.py** - Logging configuration
- **test_client.py** - Main TestClient API

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
2. Add exports to subdirectory's `__init__.py`
3. Optionally add to `utils/__init__.py` for backward compatibility

### New Subdirectory
1. Create directory with `__init__.py`
2. Add modules
3. Update `utils/__init__.py` for commonly used classes

## Testing

```bash
# Verify imports
python -c "from utils import TestClient, SystemDetector; print('OK')"

# Run tests
python build_tools/github_actions/test_executable_scripts/test_rocfft_benchmark.py
```
