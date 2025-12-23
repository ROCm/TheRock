# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

Run the setup script:

```powershell
.\setup.ps1
```

Or manually install:

```bash
pip install -r requirements.txt
```

### Step 2: Set Your API Key

```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

### Step 3: Run Analysis

```bash
python performance_analysis.py "C:\Users\rponnuru\Downloads\SubTestCountsMatrixView.csv"
```

## 📊 What You'll Get

After running the analysis, you'll get three files:

1. **performance_report.md** - AI-generated comprehensive report
2. **raw_analysis.json** - Structured data for further processing
3. **analysis_prompt.txt** - The prompt sent to the LLM (for debugging)

## 💡 Example Output

The report includes:

✅ **Executive Summary** - Overall infrastructure health  
✅ **Configuration Issues** - Configs with performance drops  
✅ **User Analysis** - Performance by user/engineer  
✅ **Hardware Issues** - Platform-specific problems  
✅ **Test Failures** - Tests failing across configs  
✅ **Recommendations** - Actionable next steps  

## 🎯 Key Features

### LangChain Framework
- Structured LLM interactions
- Token usage tracking
- Cost estimation
- Verbose logging

### Guardrails AI
- Input validation
- Output quality checks
- Topic restriction
- Professional output enforcement

### Comprehensive Analysis
- Config-specific drops
- User-specific patterns
- Hardware compatibility
- Test failure patterns

## 📈 Cost Estimates

| Model | Typical Cost |
|-------|--------------|
| gpt-4o | $0.10 - $0.50 |
| gpt-4o-mini | $0.02 - $0.10 |
| gpt-4-turbo | $0.15 - $0.70 |

## 🔧 Troubleshooting

### API Key Issues
```powershell
# Check if set
echo $env:OPENAI_API_KEY

# Set it
$env:OPENAI_API_KEY="your-key"
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### CSV Format Issues
- Ensure CSV has the required metadata columns
- Check configuration column format: `Machine | OS | Hardware | GPU | User | Deployment`

## 📚 More Examples

See `example_usage.py` for:
- Basic usage
- Custom API key
- Step-by-step analysis
- Data exploration (no API key needed)

## 🆘 Need Help?

1. Check **README.md** for detailed documentation
2. Review **example_usage.py** for code examples
3. Run with `--help` flag: `python performance_analysis.py --help`

## 🔐 Security Notes

- Never commit your API key to git
- Use environment variables or .env file
- API key is only sent to OpenAI servers
- All analysis is done securely

## 📦 What's Included

```
TheRock/
├── performance_analysis.py    # Main analysis tool
├── requirements.txt           # Dependencies
├── README.md                  # Full documentation
├── QUICKSTART.md             # This file
├── example_usage.py          # Usage examples
├── setup.ps1                 # Windows setup script
└── .env.example             # Environment template
```

## 🎓 Advanced Usage

### Different Models
```bash
python performance_analysis.py data.csv --model gpt-4o-mini
```

### Custom Output Files
```bash
python performance_analysis.py data.csv \
    --output-report my_report.md \
    --output-raw my_data.json
```

### Keep All Test Rows (Including Zeros)
By default, tests with zero executions across all configs are dropped. To keep them:
```bash
python performance_analysis.py data.csv --keep-zero-rows
```

### Programmatic Usage
```python
from performance_analysis import PerformanceAnalyzer

analyzer = PerformanceAnalyzer("data.csv", model="gpt-4o")
report, stats = analyzer.run_full_analysis()
print(f"Cost: ${stats['total_cost']:.4f}")
```

## ✨ Next Steps

1. Run your first analysis
2. Review the generated report
3. Explore the raw JSON data
4. Customize for your needs
5. Share insights with your team

---

**Happy Analyzing! 🎉**

