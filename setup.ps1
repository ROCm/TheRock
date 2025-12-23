# PowerShell setup script for Performance Analysis Tool
# Run this script to set up the environment

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Performance Analysis Tool - Setup Script" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Check pip installation
Write-Host "Checking pip installation..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>&1
    Write-Host "✓ pip found: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ pip not found! Please install pip first." -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray
Write-Host ""

pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed successfully!" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Check for API key
Write-Host ""
Write-Host "Checking for OpenAI API key..." -ForegroundColor Yellow

if ($env:OPENAI_API_KEY) {
    Write-Host "✓ OPENAI_API_KEY environment variable is set" -ForegroundColor Green
} else {
    Write-Host "⚠ OPENAI_API_KEY environment variable is NOT set" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To set your API key, run:" -ForegroundColor Cyan
    Write-Host '  $env:OPENAI_API_KEY="your-api-key-here"' -ForegroundColor White
    Write-Host ""
    Write-Host "Or create a .env file with:" -ForegroundColor Cyan
    Write-Host '  OPENAI_API_KEY=your-api-key-here' -ForegroundColor White
}

# Setup complete
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now run the tool with:" -ForegroundColor Cyan
Write-Host '  python performance_analysis.py "path\to\your\data.csv"' -ForegroundColor White
Write-Host ""
Write-Host "For more information, see README.md" -ForegroundColor Gray
Write-Host ""

