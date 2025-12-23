# Log Analyzer - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```powershell
pip install -r requirements-log-analyzer.txt
```

### Step 2: Set Your API Key (or use Ollama for FREE!)

**Option A: OpenAI (Recommended)**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

**Option B: Mistral AI**
```powershell
$env:MISTRAL_API_KEY="your-mistral-key-here"
```

**Option C: Ollama (FREE - Runs Locally!)**
```powershell
# Install from: https://ollama.ai
ollama pull llama2
ollama serve
```

### Step 3: Analyze Your Logs

```bash
# Analyze a single log file
python log_analyzer.py your_error.log

# Analyze all logs in a directory
python log_analyzer.py ./logs --pattern "*.log"

# Use local Ollama (FREE!)
python log_analyzer.py error.log --provider ollama --model llama2
```

## 📊 What You Get

After analysis, you'll get a detailed report with:

✅ **Primary Failure Reason** - Clear explanation of what went wrong  
✅ **Error Classification** - Type of error (Runtime, Config, Network, etc.)  
✅ **Root Cause Analysis** - Deep investigation of the issue  
✅ **Recommended Actions** - Step-by-step fix instructions  
✅ **Confidence Level** - How certain the analysis is  

## 💡 Example Output

```markdown
## Primary Failure Reason
GPU memory allocation failure during BabelStream-HIP benchmark.
Test requested 128GB but only 64GB available.

## Recommended Actions
1. Reduce memory request to 64GB or less
2. Optimize memory usage in benchmark code
3. Check for memory leaks from previous tests
4. Upgrade GPU if 128GB is truly required
```

## 🎯 Common Commands

```bash
# Quick analysis with OpenAI
python log_analyzer.py error.log

# Use cheaper OpenAI model
python log_analyzer.py error.log --model gpt-4o-mini

# Analyze with Mistral
python log_analyzer.py error.log --provider mistral

# FREE local analysis with Ollama
python log_analyzer.py error.log --provider ollama --model llama2

# Analyze entire directory
python log_analyzer.py ./test_logs --pattern "*.log"

# Custom output file
python log_analyzer.py error.log --output my_analysis.md

# Save as JSON too
python log_analyzer.py error.log --output-json results.json

# Corporate network (SSL issues)
python log_analyzer.py error.log --no-ssl-verify
```

## 💰 Cost Comparison

| Provider | Model | Cost per Log | Speed | Quality |
|----------|-------|--------------|-------|---------|
| **Ollama** | llama2 | **$0.00** | Medium | Good |
| **Ollama** | mistral | **$0.00** | Medium | Good |
| **OpenAI** | gpt-4o-mini | ~$0.001 | Fast | Good |
| **OpenAI** | gpt-4o | ~$0.01 | Fast | Excellent |
| **Mistral** | mistral-small | ~$0.001 | Fast | Good |
| **Mistral** | mistral-large | ~$0.008 | Fast | Excellent |

**💡 Tip:** Use Ollama for FREE local analysis!

## 📚 More Examples

See `log_analyzer_examples.py` for:
- Single log analysis
- Multiple provider comparison
- Batch processing
- Custom configuration
- Programmatic usage

## 🛠️ Troubleshooting

**Q: "No module named 'langchain'"**
```bash
pip install -r requirements-log-analyzer.txt
```

**Q: "OpenAI API key required"**
```bash
$env:OPENAI_API_KEY="your-key"
# or use Ollama: --provider ollama
```

**Q: "Ollama connection failed"**
```bash
ollama serve
ollama pull llama2
```

**Q: SSL certificate errors?**
```bash
python log_analyzer.py error.log --no-ssl-verify
```

## 🎓 Next Steps

1. Try the example: `python log_analyzer.py example_error.log`
2. Analyze your own logs
3. Try different providers to compare results
4. Check `LOG_ANALYZER_README.md` for detailed docs
5. Explore `log_analyzer_examples.py` for more usage patterns

## 🌟 Pro Tips

1. **Use gpt-4o-mini** for cost-effective daily analysis
2. **Use Ollama** for sensitive logs (stays on your machine)
3. **Batch process** multiple logs to find common patterns
4. **Compare providers** to see which gives best insights
5. **Lower temperature** (0.1-0.3) for factual analysis

---

**Happy Debugging! 🐛 → ✅**

For full documentation, see `LOG_ANALYZER_README.md`

