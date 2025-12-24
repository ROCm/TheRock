# Log Analyzer - Architecture & Execution Flow

## 📊 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        LOG ANALYZER SYSTEM                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │  Single Log  │         │  Directory   │                     │
│  │     File     │         │  of Logs     │                     │
│  └──────┬───────┘         └──────┬───────┘                     │
│         │                        │                              │
│         └────────────┬───────────┘                              │
│                      │                                          │
└──────────────────────┼──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   COMMAND LINE INTERFACE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  • ArgumentParser (argparse)                                    │
│  • Validates inputs                                             │
│  • Parses CLI arguments:                                        │
│    - log_path (file/directory)                                  │
│    - --provider (openai/mistral/ollama/azure)                   │
│    - --model                                                    │
│    - --api-key                                                  │
│    - --output, --output-json                                    │
│    - --temperature                                              │
│    - --no-ssl-verify                                            │
│                                                                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LOG ANALYZER CLASS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              INITIALIZATION LAYER                      │    │
│  ├────────────────────────────────────────────────────────┤    │
│  │  • __init__()                                          │    │
│  │    - Set provider & model                              │    │
│  │    - Get API key from env                              │    │
│  │    - Initialize LLM client                             │    │
│  │                                                        │    │
│  │  LLM Provider Initialization:                          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │    │
│  │  │ OpenAI   │  │ Mistral  │  │  Ollama  │           │    │
│  │  │ (ChatGPT)│  │   AI     │  │ (Local)  │           │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘           │    │
│  │       │             │             │                   │    │
│  │       └─────────────┴─────────────┘                   │    │
│  │                     │                                 │    │
│  │              LangChain Client                         │    │
│  └────────────────────┬───────────────────────────────────┘    │
│                       │                                         │
│  ┌────────────────────▼───────────────────────────────────┐    │
│  │              PARSING LAYER                             │    │
│  ├────────────────────────────────────────────────────────┤    │
│  │  parse_log_file()                                      │    │
│  │                                                        │    │
│  │  1. Read log file content                             │    │
│  │  2. Extract patterns:                                  │    │
│  │     ├─ ERROR patterns                                  │    │
│  │     ├─ FATAL patterns                                  │    │
│  │     ├─ CRITICAL patterns                               │    │
│  │     ├─ Exception patterns                              │    │
│  │     ├─ WARNING patterns                                │    │
│  │     └─ Stack traces                                    │    │
│  │  3. Count error types                                  │    │
│  │  4. Extract file statistics                            │    │
│  │                                                        │    │
│  │  Output: Parsed Data Dictionary                        │    │
│  └────────────────────┬───────────────────────────────────┘    │
│                       │                                         │
│  ┌────────────────────▼───────────────────────────────────┐    │
│  │              ANALYSIS LAYER                            │    │
│  ├────────────────────────────────────────────────────────┤    │
│  │  analyze_failure()                                     │    │
│  │                                                        │    │
│  │  1. Create analysis prompt                             │    │
│  │     └─ _create_analysis_prompt()                       │    │
│  │        ├─ Error statistics                             │    │
│  │        ├─ Sample errors                                │    │
│  │        ├─ Stack traces                                 │    │
│  │        └─ Recent log snippet                           │    │
│  │                                                        │    │
│  │  2. Invoke LLM with structured prompt                  │    │
│  │     ├─ SystemMessage (expert instructions)             │    │
│  │     └─ HumanMessage (log data)                         │    │
│  │                                                        │    │
│  │  3. Parse LLM response                                 │    │
│  │     └─ _parse_analysis_response()                      │    │
│  │        ├─ Primary failure reason                       │    │
│  │        ├─ Error classification                         │    │
│  │        ├─ Root cause analysis                          │    │
│  │        ├─ Recommendations                              │    │
│  │        └─ Confidence level                             │    │
│  │                                                        │    │
│  │  Output: Analysis Results Dictionary                   │    │
│  └────────────────────┬───────────────────────────────────┘    │
│                       │                                         │
│  ┌────────────────────▼───────────────────────────────────┐    │
│  │         MULTI-LOG ANALYSIS (Optional)                  │    │
│  ├────────────────────────────────────────────────────────┤    │
│  │  analyze_multiple_logs()                               │    │
│  │                                                        │    │
│  │  1. Find all log files matching pattern                │    │
│  │  2. Analyze each file individually                     │    │
│  │  3. Find common patterns                               │    │
│  │     └─ _find_common_patterns()                         │    │
│  │        ├─ Most common error types                      │    │
│  │        ├─ Unique failure count                         │    │
│  │        └─ Failure frequency                            │    │
│  │                                                        │    │
│  │  Output: Combined Results Dictionary                   │    │
│  └────────────────────┬───────────────────────────────────┘    │
│                       │                                         │
│  ┌────────────────────▼───────────────────────────────────┐    │
│  │              REPORT GENERATION                         │    │
│  ├────────────────────────────────────────────────────────┤    │
│  │  save_analysis_report()                                │    │
│  │                                                        │    │
│  │  1. Format analysis results                            │    │
│  │  2. Generate markdown report                           │    │
│  │  3. Save to file                                       │    │
│  │                                                        │    │
│  └────────────────────┬───────────────────────────────────┘    │
│                       │                                         │
└───────────────────────┼─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │  Markdown Report │         │   JSON Results   │             │
│  │    (.md file)    │         │   (.json file)   │             │
│  └──────────────────┘         └──────────────────┘             │
│                                                                  │
│  Contains:                     Contains:                        │
│  • Summary statistics          • Structured data                │
│  • Primary failure reason      • All parsed errors              │
│  • Error classification        • Full analysis text             │
│  • Detailed analysis           • Timestamps & metadata          │
│  • Recommendations             • Provider/model info            │
│  • Root cause                  • Log statistics                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Execution Flow Diagram

### Single File Analysis Flow

```
START
  │
  ├─► Parse CLI Arguments
  │   └─► log_path, provider, model, api-key, output options
  │
  ├─► Initialize LogAnalyzer
  │   ├─► Get API key from environment
  │   ├─► Select LLM provider (OpenAI/Mistral/Ollama/Azure)
  │   └─► Initialize LangChain client
  │        ├─► Success → Continue
  │        └─► Failure → Exit with error
  │
  ├─► Check if log_path is file or directory
  │   ├─► Is File? → Continue to Single File Analysis
  │   └─► Is Directory? → Jump to Multi-File Analysis
  │
  ├─► SINGLE FILE ANALYSIS
  │   │
  │   ├─► Step 1: Parse Log File
  │   │   ├─► Read file content (UTF-8, ignore errors)
  │   │   ├─► Extract patterns using regex:
  │   │   │   ├─► ERROR: r'ERROR[:\s]+(.*)'
  │   │   │   ├─► FATAL: r'FATAL[:\s]+(.*)'
  │   │   │   ├─► CRITICAL: r'CRITICAL[:\s]+(.*)'
  │   │   │   ├─► Exception: r'Exception[:\s]+(.*)'
  │   │   │   ├─► Traceback: r'Traceback.*?(?=\n\n|\Z)'
  │   │   │   ├─► Failed: r'Failed[:\s]+(.*)'
  │   │   │   └─► AssertionError: r'AssertionError[:\s]+(.*)'
  │   │   ├─► Extract warnings: r'WARNING[:\s]+(.*)'
  │   │   ├─► Extract stack traces
  │   │   ├─► Count error types (Counter)
  │   │   ├─► Get file stats (size, line count)
  │   │   └─► Return parsed_data dictionary
  │   │
  │   ├─► Step 2: Create Analysis Prompt
  │   │   ├─► Add file metadata (path, lines, size)
  │   │   ├─► Add error statistics
  │   │   ├─► Add error type distribution
  │   │   ├─► Add sample error messages (first 10)
  │   │   ├─► Add stack traces (first 3)
  │   │   ├─► Add recent log content (last 5000 chars)
  │   │   └─► Format structured prompt
  │   │
  │   ├─► Step 3: Invoke LLM for Analysis
  │   │   ├─► Create SystemMessage (expert analyst role)
  │   │   ├─► Create HumanMessage (prompt with log data)
  │   │   ├─► Call llm.invoke(messages)
  │   │   │   ├─► Success → Parse response
  │   │   │   └─► Failure (Connection error) → Return error dict
  │   │   └─► Extract response content
  │   │
  │   ├─► Step 4: Parse LLM Response
  │   │   ├─► Extract sections using regex:
  │   │   │   ├─► Primary Failure Reason
  │   │   │   ├─► Error Type Classification
  │   │   │   ├─► Root Cause Analysis
  │   │   │   ├─► Recommended Actions
  │   │   │   └─► Confidence Level
  │   │   ├─► Add metadata (timestamp, provider, model)
  │   │   ├─► Add log statistics
  │   │   └─► Return analysis_result dictionary
  │   │
  │   ├─► Step 5: Save Reports
  │   │   ├─► Save Markdown Report
  │   │   │   ├─► Format as markdown
  │   │   │   ├─► Add sections: summary, stats, analysis
  │   │   │   └─► Write to output file
  │   │   └─► Save JSON Results (if --output-json)
  │   │       ├─► Convert to JSON
  │   │       └─► Write to output file
  │   │
  │   └─► Display completion message
  │
  └─► END


MULTI-FILE ANALYSIS (Directory)
  │
  ├─► Find all log files matching pattern (*.log)
  │   └─► Use Path(directory).glob(pattern)
  │
  ├─► Initialize results dictionary
  │   ├─► summary: {total_files, analyzed_at, provider, model}
  │   ├─► individual_analyses: []
  │   └─► common_patterns: None
  │
  ├─► FOR EACH log file:
  │   ├─► Parse log file (same as Step 1 above)
  │   ├─► Analyze failure (same as Steps 2-4 above)
  │   ├─► Append to individual_analyses[]
  │   └─► Continue to next file
  │
  ├─► Find Common Patterns Across All Logs
  │   ├─► Collect all error types
  │   ├─► Collect all primary reasons
  │   ├─► Count frequency using Counter
  │   └─► Return:
  │       ├─► most_common_error_types (top 5)
  │       ├─► total_unique_failures
  │       └─► failure_frequency (top 10)
  │
  ├─► Generate Combined Report
  │   ├─► Add summary section
  │   ├─► Add common patterns section
  │   ├─► FOR EACH individual analysis:
  │   │   ├─► Add file name
  │   │   ├─► Add primary failure reason
  │   │   ├─► Add error classification
  │   │   ├─► Add log statistics
  │   │   └─► Add detailed analysis
  │   └─► Save to output file
  │
  └─► Display completion message
```

---

## 🏗️ Component Architecture

### 1. **LogAnalyzer Class Structure**

```
LogAnalyzer
│
├── __init__(provider, model, api_key, base_url, temperature, max_tokens, verify_ssl)
│   └── Initializes LLM client based on provider
│
├── Private Helper Methods
│   ├── _get_default_model(provider) → str
│   ├── _get_api_key(provider) → Optional[str]
│   ├── _initialize_llm() → LLM Client
│   ├── _init_openai() → ChatOpenAI
│   ├── _init_mistral() → ChatMistralAI
│   ├── _init_ollama() → Ollama
│   ├── _init_azure() → ChatOpenAI
│   ├── _create_analysis_prompt(log_data) → str
│   ├── _parse_analysis_response(response_text, log_data) → Dict
│   └── _find_common_patterns(analyses) → Dict
│
└── Public Methods
    ├── parse_log_file(log_file_path) → Dict[str, Any]
    ├── analyze_failure(log_data) → Dict[str, Any]
    ├── analyze_multiple_logs(log_directory, pattern) → Dict[str, Any]
    └── save_analysis_report(analysis_results, output_file)
```

### 2. **Data Flow**

```
Input File(s)
    ↓
[parse_log_file]
    ↓
Parsed Data Dict {
    file_path: str
    file_size: int
    line_count: int
    total_errors: int
    total_warnings: int
    total_stack_traces: int
    error_types: Dict[str, int]
    errors: List[str]
    warnings: List[str]
    stack_traces: List[str]
    log_snippet: str
}
    ↓
[analyze_failure]
    ↓
Analysis Results Dict {
    timestamp: str
    provider: str
    model: str
    file_analyzed: str
    full_analysis: str
    primary_reason: str
    error_type: str
    root_cause: str
    recommendations: str
    confidence: str
    log_stats: Dict
}
    ↓
[save_analysis_report]
    ↓
Output Files:
    • report.md (Markdown)
    • results.json (JSON)
```

### 3. **LLM Provider Architecture**

```
                    ┌──────────────────┐
                    │  LogAnalyzer     │
                    │  _initialize_llm │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
    ┌───────▼──────┐  ┌──────▼─────┐  ┌──────▼─────┐
    │   OpenAI     │  │  Mistral   │  │   Ollama   │
    │   Provider   │  │  Provider  │  │  Provider  │
    └───────┬──────┘  └──────┬─────┘  └──────┬─────┘
            │                │                │
    ┌───────▼──────┐  ┌──────▼─────┐  ┌──────▼─────┐
    │ ChatOpenAI   │  │ ChatMistral│  │   Ollama   │
    │ (LangChain)  │  │ (LangChain)│  │ (LangChain)│
    └───────┬──────┘  └──────┬─────┘  └──────┬─────┘
            │                │                │
            └────────────────┴────────────────┘
                             │
                    ┌────────▼─────────┐
                    │   LLM Response   │
                    └──────────────────┘
```

### 4. **Error Handling Flow**

```
Try:
    ├─► Initialize Analyzer
    │   └─► Exception? → Print error, exit(1)
    │
    ├─► Parse Log File
    │   ├─► FileNotFoundError? → Raise exception
    │   └─► UnicodeError? → Ignore with errors='ignore'
    │
    ├─► Analyze with LLM
    │   ├─► Connection Error? → Return error dict
    │   ├─► API Error? → Return error dict
    │   └─► Success? → Parse response
    │
    └─► Save Reports
        └─► IOError? → Print error message

Catch:
    └─► Print traceback, exit(1)
```

---

## 🎯 Key Design Patterns

### 1. **Strategy Pattern**
- Multiple LLM providers with unified interface
- Selectable at runtime via `--provider` flag

### 2. **Template Method Pattern**
- `analyze_failure()` defines algorithm structure
- Subcomponents handle specific steps

### 3. **Factory Pattern**
- `_initialize_llm()` creates appropriate LLM client
- Based on provider configuration

### 4. **Builder Pattern**
- `_create_analysis_prompt()` builds structured prompts
- Assembles components progressively

---

## 📦 Dependencies Architecture

```
┌─────────────────────────────────────────┐
│         Log Analyzer Application        │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌─────────┐
│ pandas │  │langchain │  │  httpx  │
│        │  │  -core   │  │         │
└────────┘  └─────┬────┘  └─────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│langchain │  │langchain │  │langchain │
│ -openai  │  │ -mistral │  │-community│
└──────────┘  └──────────┘  └──────────┘
```

---

## 🔐 Security Architecture

```
API Key Management:
    │
    ├─► Environment Variables (Preferred)
    │   ├─► OPENAI_API_KEY
    │   ├─► MISTRAL_API_KEY
    │   └─► AZURE_OPENAI_API_KEY
    │
    ├─► Command Line (--api-key)
    │   └─► Used only if env var not set
    │
    └─► Never stored in code

SSL Verification:
    │
    ├─► Default: Enabled (verify_ssl=True)
    │
    └─► Corporate Networks: --no-ssl-verify flag
        └─► Creates httpx.Client(verify=False)
```

---

## 📈 Performance Characteristics

```
┌─────────────────────────────────────────────┐
│          Performance Profile                │
├─────────────────────────────────────────────┤
│                                             │
│  File Parsing:        O(n)                  │
│    - n = file size                          │
│    - Regex matching dominates               │
│                                             │
│  LLM Analysis:        ~2-10 seconds         │
│    - Depends on:                            │
│      • Model speed (GPT-4 vs GPT-3.5)       │
│      • Network latency                      │
│      • Token count                          │
│                                             │
│  Report Generation:   O(m)                  │
│    - m = number of errors                   │
│                                             │
│  Memory Usage:        ~50-200 MB            │
│    - Depends on log file size               │
│    - Full file loaded into memory           │
│                                             │
│  Multi-file Analysis: O(k * (n + t))        │
│    - k = number of files                    │
│    - n = parsing time per file              │
│    - t = LLM analysis time per file         │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 🌟 Key Features by Layer

### Input Layer
- ✅ Single file support
- ✅ Directory batch processing
- ✅ File pattern matching (*.log, etc.)

### Processing Layer
- ✅ Regex-based error extraction
- ✅ Pattern counting and statistics
- ✅ Stack trace detection
- ✅ Error type classification

### Analysis Layer
- ✅ Multi-LLM provider support
- ✅ Structured prompt generation
- ✅ Response parsing with regex
- ✅ Confidence assessment

### Output Layer
- ✅ Markdown report generation
- ✅ JSON data export
- ✅ Pretty formatting
- ✅ Multiple file summaries

### Cross-Cutting Concerns
- ✅ Error handling at all levels
- ✅ SSL certificate bypass option
- ✅ API key management
- ✅ Progress reporting
- ✅ Extensible provider system

---

## 🚀 Typical Use Cases

### Use Case 1: Quick Single File Analysis
```bash
python log_analyzer.py app.log --no-ssl-verify
```
**Flow:** Input → Parse → Analyze → Report (Default: log_analysis_report.md)

### Use Case 2: Batch Analysis with JSON Export
```bash
python log_analyzer.py ./logs/ --pattern "*.log" --output-json results.json
```
**Flow:** Input Directory → Find Files → Parse Each → Analyze Each → Find Patterns → Combined Report

### Use Case 3: Different LLM Provider
```bash
python log_analyzer.py error.log --provider mistral --model mistral-large-latest
```
**Flow:** Input → Initialize Mistral → Parse → Analyze with Mistral → Report

---


