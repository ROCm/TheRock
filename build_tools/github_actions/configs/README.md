# Test Framework

Automated benchmark testing framework for ROCm libraries with system detection, results collection, and performance tracking.

## Features

- Automated benchmark execution (ROCfft, ROCrand, ROCsolver, hipBLASLt)
- Hardware, OS, GPU, and ROCm auto-detection
- Local storage (JSON) and API upload
- LKG (Last Known Good) comparison
- File rotation and configurable logging
- Modular, extensible architecture


## Configuration

```bash
# Required environment variables
export THEROCK_BIN_DIR=/path/to/rocm/bin
export ARTIFACT_RUN_ID=$WORKFLOW_RUN_ID
export AMDGPU_FAMILIES=gfx950-dcgpu
```

### Run Benchmarks

```bash
python build_tools/github_actions/test_executable_scripts/test_rocfft_benchmark.py
python build_tools/github_actions/test_executable_scripts/test_rocrand_benchmark.py
```

## Project Structure

```
build_tools/github_actions/
├── configs/
│   ├── config.yml              # Main configuration
│   └── benchmarks/             # Benchmark configs
│
├── test_executable_scripts/    # Benchmark scripts
│   ├── test_rocfft_benchmark.py
│   ├── test_rocrand_benchmark.py
│   ├── test_rocsolver_benchmark.py
│   └── test_hipblaslt_benchmark.py
│
└── utils/                      # Framework utilities
    ├── test_client.py          # Main client API
    ├── config/                 # Configuration management
    ├── system/                 # System detection
    └── results/                # Results handling
```

## Configuration

### Main Config: `configs/config.yml`

```yaml
Config:
  Core:
    LogLevel: INFO                    # DEBUG, INFO, WARNING, ERROR
    LogToFile: true
    LogDirectory: "./logs"
    UploadTestResultsToAPI: true
    
  Results:
    OutputDirectory: "./results"
    SaveJSON: true
```

## Architecture

### Core Components

- **TestClient** - Main API for test execution and results management
- **System Detection** - Hardware, OS, and ROCm detection
- **Results Handling** - Local storage and API submission with retry
- **Configuration** - YAML-based config with environment variable expansion

### Test Flow

1. Initialize TestClient → Detect system and load config
2. Run Benchmarks → Execute binary and capture output
3. Parse Results → Extract metrics from log file
4. Upload Results → Submit to API and save locally
5. Compare with LKG → Fetch and compare scores
6. Report Results → Display table and return status

## Adding New Benchmarks

### 1. Create Script

```python
from utils import TestClient
from utils.logger import log

def run_benchmarks():
    """Run benchmarks and save output to log file."""
    pass

def parse_results():
    """Parse benchmark results from log file."""
    pass

def main():
    client = TestClient(auto_detect=True)
    client.print_system_summary()
    run_benchmarks()
    test_results, table = parse_results()
    client.upload_results(...)

if __name__ == '__main__':
    main()
```

### 2. Add Config (Optional)

Create `configs/benchmarks/your_benchmark.json`:

```json
{
    "test_cases": ["case1", "case2"]
}
```

## Documentation

- [Main framework documentation](README.md) - Main framework documentation
- [Utils Module](../utils/README.md) - Framework utilities
- [Configuration Guide](config.yml) - Configuration options

