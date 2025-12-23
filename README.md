# Performance Drop Analysis Tool

A sophisticated performance analysis tool powered by **LangChain** and **Guardrails AI** to analyze test infrastructure performance data and identify configuration-specific and user-specific performance drops.

## Features

✨ **LangChain Integration**
- Uses LangChain framework for robust LLM interactions
- Structured prompts with system and human message templates
- Token usage tracking and cost estimation
- Verbose logging for transparency

🛡️ **Guardrails AI Protection**
- Input validation to ensure data integrity
- Output validation to prevent toxic language
- Topic restriction to maintain focus on performance analysis
- Professional and actionable output enforcement

📊 **Comprehensive Analysis**
- Configuration-specific performance drops
- User-specific performance patterns
- Hardware/OS compatibility issues
- Test failure pattern identification
- Actionable recommendations

## Installation

### 1. Clone or navigate to the repository

```bash
cd TheRock
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your OpenAI API key

**Option A: Environment Variable (Recommended)**
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Windows CMD
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

**Option B: Use .env file**
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-api-key-here
```

**Option C: Pass as argument**
Use the `--api-key` flag when running the script.

## Usage

### Basic Usage

```bash
python performance_analysis.py path/to/your/data.csv
```

### With Custom Options

```bash
python performance_analysis.py path/to/your/data.csv \
    --model gpt-4o \
    --output-report my_report.md \
    --output-raw my_analysis.json
```

### Keep All Rows (Including Zero-Only Tests)

By default, the tool drops test rows that have zero tests across ALL configurations to focus on meaningful data. To keep all rows:

```bash
python performance_analysis.py path/to/your/data.csv --keep-zero-rows
```

### Example with the provided data

```bash
python performance_analysis.py "C:\Users\rponnuru\Downloads\SubTestCountsMatrixView.csv"
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `csv_file` | Path to CSV file (required) | - |
| `--api-key` | OpenAI API key | Uses `OPENAI_API_KEY` env var |
| `--model` | OpenAI model to use | `gpt-4o` |
| `--output-report` | Output markdown report file | `performance_report.md` |
| `--output-raw` | Output raw JSON data file | `raw_analysis.json` |
| `--keep-zero-rows` | Keep test rows with zero tests across all configs | `False` (drops by default) |

### Available Models

- `gpt-4o` (recommended) - Most capable, balanced cost
- `gpt-4o-mini` - Faster, lower cost
- `gpt-4-turbo` - High capability
- `gpt-3.5-turbo` - Fastest, lowest cost

## Output Files

The tool generates three files:

1. **performance_report.md** - AI-generated analysis report with:
   - Executive summary
   - Configuration-specific issues
   - User-specific performance analysis
   - Hardware/platform issues
   - Test-specific failures
   - Actionable recommendations

2. **raw_analysis.json** - Structured data including:
   - Performance metrics per configuration
   - User performance statistics
   - Hardware performance data
   - Test failure details

3. **analysis_prompt.txt** - The exact prompt sent to the LLM (for debugging)

## CSV File Format

The tool expects a CSV file with:

### Metadata Columns:
- `Features`
- `Test_Category`
- `Test_Name`
- `Software_Features`
- `Hardware_Features`
- `Test_Area`
- `Test_Execution_Mode`
- `Test_Plan_Name`
- `Execution_Label`

### Configuration Columns:
Each configuration column should follow the format:
```
Machine | OS | Hardware | GPU_Count | User | Deployment_Type
```

Example:
```
banff-123 | Ubuntu-22.04.5 | MI300X-O | 8x | John Doe | baremetal
```

## Guardrails Features

The tool implements several guardrails to ensure quality output:

### Input Validation
- Checks for required fields
- Validates data ranges
- Prevents processing of invalid data

### Output Validation
- **Toxic Language Detection** - Ensures professional language
- **Topic Restriction** - Keeps analysis focused on performance
- **Format Validation** - Ensures structured, actionable output

### Guardrails Installation (Optional but Recommended)

If guardrails are not installed, the tool will run without them but with a warning:

```bash
pip install guardrails-ai
```

## Architecture

```
┌─────────────────────────────────────────────┐
│         Performance Analyzer                 │
│                                              │
│  1. Data Loading (pandas)                   │
│  2. Performance Analysis                     │
│  3. Test Failure Identification              │
│  4. Input Validation (Guardrails)           │
│  5. LLM Analysis (LangChain)                │
│  6. Output Validation (Guardrails)          │
│  7. Report Generation                        │
└─────────────────────────────────────────────┘
```

## Key Components

### PerformanceAnalyzer Class
Main class that orchestrates the analysis pipeline:
- Loads and parses CSV data
- Performs statistical analysis
- Integrates with LangChain for AI analysis
- Validates inputs and outputs

### PerformanceGuardrails Class
Handles all validation and safety checks:
- Input data validation
- Output quality assurance
- Topic restriction
- Toxic language prevention

### LangChain Integration
- Uses `ChatOpenAI` for LLM interactions
- Structured prompts with `ChatPromptTemplate`
- Token tracking with callbacks
- Cost estimation

## Analysis Sections

The generated report includes:

### 1. Executive Summary
Overall health and key concerns

### 2. Configuration-Specific Issues
- Configs with zero tests
- Low-performance configurations
- Common failure patterns

### 3. User-Specific Analysis
- Top and bottom performers
- Capacity issues
- Workload distribution recommendations

### 4. Hardware/Platform Issues
- Hardware-specific problems
- OS compatibility issues
- GPU configuration problems

### 5. Test-Specific Failures
- Tests failing across multiple configs
- Platform-specific failures

### 6. Actionable Recommendations
- Immediate actions
- Long-term improvements
- Resource allocation suggestions

## Troubleshooting

### "OpenAI API key not provided"
- Set the `OPENAI_API_KEY` environment variable
- Or use the `--api-key` argument

### "Guardrails AI not installed"
- The tool will still work without guardrails
- Install with: `pip install guardrails-ai`

### "CSV file not found"
- Check the file path
- Use quotes around paths with spaces
- Use absolute path if relative path doesn't work

### High token usage/cost
- Use `gpt-4o-mini` model for lower cost
- The tool displays estimated cost after analysis

## Cost Estimation

Typical analysis costs (approximate):
- **gpt-4o**: $0.10 - $0.50 per analysis
- **gpt-4o-mini**: $0.02 - $0.10 per analysis
- **gpt-4-turbo**: $0.15 - $0.70 per analysis

Actual costs depend on data size and configuration count.

## Example Output

```bash
======================================================================
Performance Drop Analysis Tool
Powered by LangChain + Guardrails
======================================================================
Loading data from SubTestCountsMatrixView.csv...
✓ Loaded 350 test cases across 118 configurations

Analyzing performance drops...
✓ Found 15 configs with zero tests
✓ Found 42 configs with low performance

Identifying test failures...
✓ Found 127 tests with high failure rates

✓ Raw analysis data saved to: raw_analysis.json

======================================================================
Sending data to LLM for analysis...
Model: gpt-4o
======================================================================

✓ Analysis complete
  - Tokens used: 3542
  - Estimated cost: $0.1245

✓ Output validated by guardrails
✓ Report saved to: performance_report.md

======================================================================
Analysis Complete!
======================================================================

Generated files:
  1. performance_report.md - AI-generated analysis report
  2. raw_analysis.json - Raw analysis data (JSON)
```

## Contributing

This tool is part of TheRock performance tracking initiative.

## License

Internal AMD use.

## Support

For issues or questions, contact the TheRock team.
