# Log Analyzer - Multi-LLM Failure Analysis Tool

Analyze log files using AI to automatically identify failure reasons, error patterns, and get actionable recommendations.

## 🌟 Features

### Multi-Provider Support
- ✅ **OpenAI** (GPT-4, GPT-4o, GPT-3.5-turbo)
- ✅ **Mistral AI** (mistral-large, mistral-medium, mistral-small)
- ✅ **Ollama** (local models: llama2, mistral, codellama, etc.) - **FREE!**
- ✅ **Azure OpenAI** (enterprise deployments)

### Intelligent Analysis
- ✅ Automatic error pattern detection
- ✅ Stack trace analysis
- ✅ Root cause identification
- ✅ Actionable recommendations
- ✅ Error type classification
- ✅ Confidence level assessment

### Flexible Usage
- ✅ Single log file analysis
- ✅ Bulk directory analysis
- ✅ Multiple provider comparison
- ✅ CLI and programmatic interfaces
- ✅ Markdown and JSON output

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements-log-analyzer.txt
```

### 2. Set Up Your LLM Provider

#### OpenAI (Default)
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

#### Mistral AI
```bash
# Get your API key from: https://console.mistral.ai/
$env:MISTRAL_API_KEY="your-mistral-key-here"
```

#### Ollama (Local - FREE!)
```bash
# Install Ollama from: https://ollama.ai
ollama pull llama2
ollama serve
```

## 🚀 Quick Start

### Command Line

```bash
# Analyze a single log file with OpenAI
python log_analyzer.py example_error.log

# Analyze with Mistral AI
python log_analyzer.py example_error.log --provider mistral

# Analyze with local Ollama (FREE!)
python log_analyzer.py example_error.log --provider ollama --model llama2

# Analyze all logs in a directory
python log_analyzer.py ./logs --pattern "*.log"

# Custom output file
python log_analyzer.py error.log --output my_analysis.md
```

### Programmatic Usage

```python
from log_analyzer import LogAnalyzer

# Initialize analyzer
analyzer = LogAnalyzer(
    provider="openai",
    model="gpt-4o-mini"
)

# Parse log file
log_data = analyzer.parse_log_file("example_error.log")

# Analyze failures
analysis = analyzer.analyze_failure(log_data)

# Save report
analyzer.save_analysis_report(analysis, "analysis_report.md")

print(f"Primary Reason: {analysis['primary_reason']}")
```

## 📖 Detailed Usage

### OpenAI Provider

```bash
# Use GPT-4o-mini (cost-effective)
python log_analyzer.py error.log --provider openai --model gpt-4o-mini

# Use GPT-4 (most capable)
python log_analyzer.py error.log --provider openai --model gpt-4o

# Custom temperature for more creative analysis
python log_analyzer.py error.log --temperature 0.7
```

**Cost Estimates:**
- gpt-4o-mini: ~$0.001-0.005 per log
- gpt-4o: ~$0.005-0.02 per log
- gpt-3.5-turbo: ~$0.0005-0.002 per log

### Mistral AI Provider

```bash
# Use Mistral Large (most capable)
python log_analyzer.py error.log --provider mistral --model mistral-large-latest

# Use Mistral Medium (balanced)
python log_analyzer.py error.log --provider mistral --model mistral-medium

# Use Mistral Small (fast & cheap)
python log_analyzer.py error.log --provider mistral --model mistral-small
```

**Cost Estimates:**
- mistral-large: ~$0.008 per log
- mistral-medium: ~$0.003 per log
- mistral-small: ~$0.001 per log

### Ollama (Local - FREE!)

```bash
# First, install and start Ollama
ollama pull llama2
ollama serve

# Then analyze logs locally (no API costs!)
python log_analyzer.py error.log --provider ollama --model llama2

# Try other models
python log_analyzer.py error.log --provider ollama --model mistral
python log_analyzer.py error.log --provider ollama --model codellama
```

**Cost: $0.00** - Runs entirely on your machine!

### Analyze Multiple Logs

```bash
# Analyze all .log files in a directory
python log_analyzer.py ./logs --pattern "*.log"

# Analyze specific pattern
python log_analyzer.py ./test_results --pattern "error_*.txt"

# Save as both markdown and JSON
python log_analyzer.py ./logs \
    --output analysis.md \
    --output-json analysis.json
```

## 🎯 Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `log_path` | Path to log file or directory | Required |
| `--provider` | LLM provider (openai/mistral/ollama/azure) | `openai` |
| `--model` | Model name (provider-specific) | Provider default |
| `--api-key` | API key for the provider | From environment |
| `--base-url` | Custom base URL (for Ollama/Azure) | Provider default |
| `--pattern` | File pattern for directory analysis | `*.log` |
| `--output` | Output markdown report file | `log_analysis_report.md` |
| `--output-json` | Also save results as JSON | None |
| `--temperature` | Model temperature (0.0-1.0) | `0.3` |
| `--no-ssl-verify` | Disable SSL verification | False |

## 📊 Output

### Markdown Report

The tool generates a comprehensive markdown report with:

1. **Primary Failure Reason** - Clear, concise explanation
2. **Error Classification** - Type of error (Config, Runtime, Network, etc.)
3. **Log Statistics** - Error counts, warnings, stack traces
4. **Root Cause Analysis** - Detailed investigation
5. **Recommended Actions** - Step-by-step fix instructions
6. **Confidence Level** - How certain the analysis is

### JSON Output

For programmatic processing:

```json
{
  "timestamp": "2024-12-23T10:15:00",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "primary_reason": "GPU memory allocation failure",
  "error_type": "Runtime Error - Memory",
  "root_cause": "Test requested 128GB but only 64GB available",
  "recommendations": "Reduce batch size or use memory-efficient mode",
  "confidence": "High",
  "log_stats": {
    "total_errors": 5,
    "total_warnings": 1,
    "total_stack_traces": 1
  }
}
```

## 💡 Usage Examples

### Example 1: Quick Single Log Analysis

```bash
python log_analyzer.py test_failure.log
```

**Output:** `log_analysis_report.md` with complete analysis

### Example 2: Compare Different Providers

```bash
# Analyze with OpenAI
python log_analyzer.py error.log --output openai_analysis.md

# Analyze with Mistral
python log_analyzer.py error.log --provider mistral --output mistral_analysis.md

# Analyze with local Ollama
python log_analyzer.py error.log --provider ollama --output ollama_analysis.md
```

**Compare the results to see which provider works best for your logs!**

### Example 3: Batch Process Test Logs

```bash
# Analyze all test logs from a run
python log_analyzer.py ./test_results_2024-12-23 \
    --pattern "test_*.log" \
    --output batch_analysis.md \
    --output-json batch_analysis.json
```

### Example 4: Corporate Network (SSL Issues)

```bash
python log_analyzer.py error.log --no-ssl-verify
```

### Example 5: Programmatic Integration

```python
from log_analyzer import LogAnalyzer
import json

# Analyze multiple logs programmatically
analyzer = LogAnalyzer(provider="openai", model="gpt-4o-mini")

test_logs = ["test1.log", "test2.log", "test3.log"]
results = []

for log_file in test_logs:
    log_data = analyzer.parse_log_file(log_file)
    analysis = analyzer.analyze_failure(log_data)
    results.append({
        'file': log_file,
        'reason': analysis.get('primary_reason'),
        'type': analysis.get('error_type')
    })

# Save summary
with open('summary.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## 🔍 What Gets Analyzed

The tool automatically detects and analyzes:

- **Error Messages** - All ERROR, FATAL, CRITICAL logs
- **Exceptions** - Python, C++, Java exceptions
- **Stack Traces** - Full traceback analysis
- **Warnings** - Important warnings that might indicate issues
- **Error Patterns** - Common error types and frequencies
- **Context** - Surrounding log entries for better understanding

## 🎨 Provider Comparison

| Provider | Cost | Speed | Quality | Local | Best For |
|----------|------|-------|---------|-------|----------|
| **OpenAI GPT-4o** | $$ | Fast | Excellent | No | Production analysis |
| **OpenAI GPT-4o-mini** | $ | Very Fast | Good | No | Cost-effective daily use |
| **Mistral Large** | $$ | Fast | Excellent | No | Alternative to GPT-4 |
| **Mistral Small** | $ | Very Fast | Good | No | Budget-friendly option |
| **Ollama (llama2)** | FREE | Medium | Good | Yes | Privacy, no API costs |
| **Ollama (mistral)** | FREE | Medium | Good | Yes | Privacy, no API costs |

## 🔒 Security & Privacy

### Data Privacy
- **OpenAI/Mistral**: Logs are sent to provider APIs for analysis
- **Ollama**: Everything runs locally, no data leaves your machine

### Best Practices
- ✅ Remove sensitive data from logs before analysis
- ✅ Use Ollama for sensitive/confidential logs
- ✅ Store API keys in environment variables, not code
- ✅ Use `--no-ssl-verify` only in trusted networks

## 🛠️ Troubleshooting

### "No module named 'langchain'"

```bash
pip install -r requirements-log-analyzer.txt
```

### "OpenAI API key required"

```bash
$env:OPENAI_API_KEY="your-api-key-here"
```

### "Mistral support not installed"

```bash
pip install langchain-mistralai
```

### "Ollama connection failed"

```bash
# Make sure Ollama is running
ollama serve

# Pull the model you want
ollama pull llama2
```

### SSL Certificate Errors

```bash
python log_analyzer.py error.log --no-ssl-verify
```

## 📚 Example Analysis Output

```markdown
# Log Analysis Report

**Generated:** 2024-12-23 16:45:00
**Provider:** openai
**Model:** gpt-4o-mini

---

## Primary Failure Reason

GPU memory allocation failure during BabelStream-HIP benchmark execution.
The test requested 128GB of GPU memory, but only 64GB was available on the MI300X configuration.

## Error Classification

Runtime Error - Memory Allocation

## Log Statistics

- Total Errors: 5
- Total Warnings: 1
- Stack Traces: 1

### Error Type Distribution:
- RuntimeError: 1
- HIPError: 1

## Detailed Analysis

The failure occurred during the BabelStream-HIP benchmark when attempting to allocate 
GPU memory. The specific error "hipErrorOutOfMemory (2)" indicates that the requested 
memory size (128GB) exceeds the available memory (64GB) on the MI300X GPU.

This suggests either:
1. The test configuration is incorrect for this hardware
2. Memory wasn't properly freed from previous tests
3. The benchmark is using an inappropriate memory size

The retry attempt with reduced memory footprint also failed, indicating a persistent 
memory availability issue.

## Recommended Actions

1. **Immediate**: Verify MI300X has 64GB memory per GPU and adjust test configuration
2. **High**: Implement proper memory cleanup between test runs
3. **Medium**: Add memory availability checks before benchmark execution
4. **Medium**: Configure BabelStream to use adaptive memory sizing
5. **Low**: Add better error messaging for memory allocation failures

## Confidence Level

High - The error message clearly indicates a memory allocation issue with specific 
size values and error codes.
```

## 🎯 Tips for Best Results

1. **Include full logs** - More context = better analysis
2. **Use appropriate model** - GPT-4o for complex issues, GPT-4o-mini for simple ones
3. **Lower temperature** - Use 0.1-0.3 for factual analysis
4. **Batch processing** - Analyze multiple logs to find patterns
5. **Compare providers** - Different LLMs might catch different issues

## 📞 Support

For issues or questions:
1. Check `log_analyzer_examples.py` for usage examples
2. Run `python log_analyzer.py --help` for all options
3. See main README.md for general setup

## 🚀 Next Steps

1. Install dependencies: `pip install -r requirements-log-analyzer.txt`
2. Set up your preferred LLM provider
3. Try the example: `python log_analyzer.py example_error.log`
4. Analyze your own logs!

---

**Happy Debugging! 🐛 → ✅**

