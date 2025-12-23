# Cost Control Guide

## Report Generation Options

The tool now gives you control over which reports to generate, allowing you to save costs when needed.

## 📊 Report Types

### 1. Test-Specific Report (FREE ✓)
- **Cost**: $0.00
- **Generation Time**: ~2-5 seconds
- **Content**:
  - Detailed pass/fail ratios for each test
  - List of configs where test passed vs failed
  - Pattern analysis (OS, hardware trends)
  - Performance drop percentages
  - Clear tabular format

### 2. AI-Powered Report (💰 ~$0.003)
- **Cost**: ~$0.003 with gpt-4o-mini, ~$0.01 with gpt-4o
- **Generation Time**: ~10-30 seconds
- **Content**:
  - Executive summary with insights
  - Root cause hypothesis
  - Semantic pattern recognition
  - Natural language recommendations
  - Cross-correlation analysis

### 3. Raw JSON Data (FREE ✓)
- **Cost**: $0.00
- **Always Generated**
- **Content**:
  - All numerical data
  - Config lists
  - Performance metrics
  - Can be processed by other tools

## 💰 Cost Comparison

| Option | Command | Reports Generated | Cost | Use Case |
|--------|---------|-------------------|------|----------|
| **Both** (Default) | `python performance_analysis.py data.csv` | Test Report + AI Report + JSON | ~$0.003 | Complete analysis with insights |
| **Test Only** (FREE!) | `python performance_analysis.py data.csv --no-ai-report` | Test Report + JSON | $0.00 | Detailed metrics without AI |
| **AI Only** | `python performance_analysis.py data.csv --report-type ai-only` | AI Report + JSON | ~$0.003 | Just high-level insights |
| **None** | `python performance_analysis.py data.csv --report-type none` | JSON only | $0.00 | Raw data for custom processing |

## 🚀 Command Examples

### Full Analysis (Both Reports)
```bash
python performance_analysis.py "C:\Users\rponnuru\Downloads\SubTestCountsMatrixView.csv"
```
**Cost**: ~$0.003  
**Output**: test_specific.md + ai_report.md + raw_analysis.json

---

### Cost-Free Mode (Test Report Only)
```bash
python performance_analysis.py "C:\Users\rponnuru\Downloads\SubTestCountsMatrixView.csv" --no-ai-report
```
**Cost**: $0.00  
**Output**: test_specific.md + raw_analysis.json  
**Note**: No API key needed!

---

### AI Insights Only
```bash
python performance_analysis.py "C:\Users\rponnuru\Downloads\SubTestCountsMatrixView.csv" --report-type ai-only
```
**Cost**: ~$0.003  
**Output**: ai_report.md + raw_analysis.json

---

### Raw Data Only
```bash
python performance_analysis.py "C:\Users\rponnuru\Downloads\SubTestCountsMatrixView.csv" --report-type none
```
**Cost**: $0.00  
**Output**: raw_analysis.json only

---

### Using Cheaper Model
```bash
python performance_analysis.py data.csv --model gpt-4o-mini
```
**Cost**: ~$0.003 (5x cheaper than gpt-4o)

---

## 📈 Cost Breakdown (350 tests, 120 configs)

| Model | Test Report | AI Report | Total Cost |
|-------|-------------|-----------|------------|
| **gpt-4o-mini** | $0.00 | ~$0.003 | ~$0.003 |
| **gpt-4o** | $0.00 | ~$0.015 | ~$0.015 |
| **Test-only** | $0.00 | $0.00 | **$0.00** |

## 💡 Recommendations

### When to Use Each Option

**Use BOTH reports when:**
- ✓ First time analyzing your data
- ✓ Need comprehensive insights
- ✓ Want AI to identify hidden patterns
- ✓ Cost is not a concern (~$0.003 is negligible)

**Use TEST-ONLY report when:**
- ✓ Regular daily/weekly monitoring
- ✓ Just need pass/fail metrics
- ✓ Want to save costs (FREE!)
- ✓ Processing many datasets
- ✓ No API key available

**Use AI-ONLY report when:**
- ✓ Need executive summary only
- ✓ Don't need detailed test breakdowns
- ✓ Want natural language insights
- ✓ Presenting to management

**Use RAW DATA only when:**
- ✓ Integrating with other tools
- ✓ Building custom dashboards
- ✓ Automated pipelines
- ✓ Want maximum flexibility

## 🎓 Usage Examples in Code

### Example 1: Cost-Free Analysis
```python
from performance_analysis import PerformanceAnalyzer

analyzer = PerformanceAnalyzer("data.csv")
report, _ = analyzer.run_full_analysis(
    generate_test_report=True,   # FREE
    generate_ai_report=False     # Skip to save costs
)
# Cost: $0.00
```

### Example 2: Full Analysis
```python
analyzer = PerformanceAnalyzer("data.csv", api_key="your-key")
report, stats = analyzer.run_full_analysis(
    generate_test_report=True,   # FREE
    generate_ai_report=True      # ~$0.003
)
print(f"Total cost: ${stats['total_cost']:.4f}")
```

### Example 3: Weekly Monitoring (Cost-Free)
```python
# Run daily without AI costs
for dataset in weekly_datasets:
    analyzer = PerformanceAnalyzer(dataset)
    analyzer.run_full_analysis(
        generate_test_report=True,
        generate_ai_report=False  # Save costs on daily runs
    )
# Monthly AI analysis for trends
analyzer.run_full_analysis(
    generate_test_report=True,
    generate_ai_report=True  # Deep insights once a month
)
```

## 📊 What You Get in Each Report

### Test-Specific Report (FREE)
```markdown
## Test: BabelStream-HIP
**Pass Ratio**: 29.17% (35/120 configs)
**Fail Ratio**: 70.83% (85/120 configs)

### ✓ Configs Where Test PASSED
- Config A: Ubuntu 22.04, MI300X, User X
- Config B: RHEL 9.6, MI200, User Y
...

### ✗ Configs Where Test FAILED
- Config C: CentOS Stream 9, Navi48, User Z
- Config D: Ubuntu 24.04, Navi32, User W
...

### Pattern Analysis
Most Common OS in Failures: CentOS-Stream-9 (45%)
Most Common Hardware in Failures: MI300X-O (60%)
```

### AI-Powered Report (~$0.003)
```markdown
## Executive Summary
The MI300X series shows 65% failure rate on RHEL 9.7+,
suggesting a systematic driver incompatibility...

## Root Cause Hypothesis
All MI300-series GPUs fail on RHEL 9.7+ due to kernel 5.14+
incompatibility. This pattern doesn't appear on RHEL 9.6...

## Recommendations
1. IMMEDIATE: Block RHEL 9.7 adoption for MI300X
2. HIGH: Investigate MI300X driver on kernel 5.14+
...
```

## 🎯 Best Practices

1. **Development/Testing**: Use `--no-ai-report` to save costs
2. **Production Monitoring**: Run test-only reports daily, AI report weekly
3. **Investigations**: Use both reports for deep dives
4. **Presentations**: Generate AI report for stakeholder updates
5. **Automation**: Use `--report-type none` for pipeline integration

## ❓ FAQ

**Q: Can I generate test report without AI report?**  
A: Yes! Use `--no-ai-report` flag. Cost: $0.00

**Q: Do I need an API key for test reports?**  
A: No! Test reports are generated locally without API calls.

**Q: How much does it cost to analyze 1000 tests?**  
A: Test report: $0.00, AI report: ~$0.008 with gpt-4o-mini

**Q: Can I run this daily without costs?**  
A: Yes! Use `--no-ai-report` for $0.00 daily monitoring.

**Q: What's the difference in quality?**  
A: Test report gives you data/metrics. AI report gives you insights/recommendations.

## 📞 Support

For questions or issues, check the main README.md or raise an issue on GitHub.

