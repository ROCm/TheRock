"""
Log Analyzer - Usage Examples

This file demonstrates various ways to use the Log Analyzer tool
with different LLM providers.
"""

import os
from log_analyzer import LogAnalyzer


def example_openai_single_log():
    """Example 1: Analyze a single log file with OpenAI"""
    print("\n" + "="*70)
    print("Example 1: Single Log Analysis with OpenAI")
    print("="*70)
    
    # Initialize with OpenAI (default)
    analyzer = LogAnalyzer(
        provider="openai",
        model="gpt-4o-mini",  # Cost-effective option
        verify_ssl=False  # For corporate networks
    )
    
    # Parse log file
    log_data = analyzer.parse_log_file("example_error.log")
    
    # Analyze failures
    analysis = analyzer.analyze_failure(log_data)
    
    # Save report
    analyzer.save_analysis_report(analysis, "openai_analysis.md")
    
    print("\n[OK] Analysis complete!")
    print(f"Primary Reason: {analysis.get('primary_reason', 'N/A')}")


def example_mistral_single_log():
    """Example 2: Analyze with Mistral AI"""
    print("\n" + "="*70)
    print("Example 2: Single Log Analysis with Mistral")
    print("="*70)
    
    # Check if Mistral API key is set
    if not os.getenv('MISTRAL_API_KEY'):
        print("[WARNING] MISTRAL_API_KEY not set. Skipping example.")
        return
    
    analyzer = LogAnalyzer(
        provider="mistral",
        model="mistral-large-latest"  # Most capable Mistral model
    )
    
    log_data = analyzer.parse_log_file("example_error.log")
    analysis = analyzer.analyze_failure(log_data)
    analyzer.save_analysis_report(analysis, "mistral_analysis.md")
    
    print("[OK] Mistral analysis complete!")


def example_ollama_local():
    """Example 3: Analyze with local Ollama model (FREE!)"""
    print("\n" + "="*70)
    print("Example 3: Local Analysis with Ollama (FREE)")
    print("="*70)
    
    # No API key needed - runs locally!
    try:
        analyzer = LogAnalyzer(
            provider="ollama",
            model="llama2",  # or "mistral", "codellama", etc.
            base_url="http://localhost:11434"
        )
        
        log_data = analyzer.parse_log_file("example_error.log")
        analysis = analyzer.analyze_failure(log_data)
        analyzer.save_analysis_report(analysis, "ollama_analysis.md")
        
        print("[OK] Local analysis complete! No API costs! 🎉")
        
    except Exception as e:
        print(f"[ERROR] Ollama not available: {e}")
        print("To use Ollama:")
        print("  1. Install: https://ollama.ai")
        print("  2. Pull model: ollama pull llama2")
        print("  3. Run server: ollama serve")


def example_multiple_logs():
    """Example 4: Analyze multiple log files in a directory"""
    print("\n" + "="*70)
    print("Example 4: Multiple Log Analysis")
    print("="*70)
    
    analyzer = LogAnalyzer(
        provider="openai",
        model="gpt-4o-mini"
    )
    
    # Analyze all .log files in a directory
    results = analyzer.analyze_multiple_logs(
        log_directory="./logs",
        pattern="*.log"
    )
    
    # Save combined report
    analyzer.save_analysis_report(results, "multi_log_analysis.md")
    
    print(f"\n[OK] Analyzed {results['summary']['total_files']} files")


def example_custom_configuration():
    """Example 5: Custom configuration for specific needs"""
    print("\n" + "="*70)
    print("Example 5: Custom Configuration")
    print("="*70)
    
    analyzer = LogAnalyzer(
        provider="openai",
        model="gpt-4o",  # Most capable model
        temperature=0.1,  # Very focused/deterministic
        max_tokens=6000,  # Longer responses
        verify_ssl=False
    )
    
    log_data = analyzer.parse_log_file("complex_error.log")
    analysis = analyzer.analyze_failure(log_data)
    
    # Save both markdown and JSON
    analyzer.save_analysis_report(analysis, "custom_analysis.md")
    
    import json
    with open("custom_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print("[OK] Custom analysis with JSON output complete!")


def example_batch_processing():
    """Example 6: Process multiple logs from different test runs"""
    print("\n" + "="*70)
    print("Example 6: Batch Processing Test Logs")
    print("="*70)
    
    analyzer = LogAnalyzer(provider="openai", model="gpt-4o-mini")
    
    # List of log files from different test runs
    test_logs = [
        "test_run_1.log",
        "test_run_2.log",
        "test_run_3.log"
    ]
    
    all_analyses = []
    
    for log_file in test_logs:
        if os.path.exists(log_file):
            print(f"\n[*] Analyzing {log_file}...")
            log_data = analyzer.parse_log_file(log_file)
            analysis = analyzer.analyze_failure(log_data)
            all_analyses.append({
                'file': log_file,
                'analysis': analysis
            })
    
    # Create combined report
    combined_results = {
        'summary': {
            'total_files': len(all_analyses),
            'provider': 'openai',
            'model': 'gpt-4o-mini'
        },
        'individual_analyses': all_analyses
    }
    
    analyzer.save_analysis_report(combined_results, "batch_analysis.md")
    print(f"\n[OK] Batch processing complete! Analyzed {len(all_analyses)} logs")


def example_compare_providers():
    """Example 7: Compare analysis from different providers"""
    print("\n" + "="*70)
    print("Example 7: Compare Different LLM Providers")
    print("="*70)
    
    log_file = "example_error.log"
    
    if not os.path.exists(log_file):
        print(f"[ERROR] {log_file} not found")
        return
    
    providers = [
        ("openai", "gpt-4o-mini"),
        ("mistral", "mistral-large-latest") if os.getenv('MISTRAL_API_KEY') else None,
    ]
    
    providers = [p for p in providers if p is not None]
    
    for provider, model in providers:
        print(f"\n--- Analyzing with {provider}/{model} ---")
        
        try:
            analyzer = LogAnalyzer(provider=provider, model=model)
            log_data = analyzer.parse_log_file(log_file)
            analysis = analyzer.analyze_failure(log_data)
            
            output_file = f"{provider}_comparison.md"
            analyzer.save_analysis_report(analysis, output_file)
            
            print(f"[OK] {provider} analysis saved to {output_file}")
            
        except Exception as e:
            print(f"[ERROR] {provider} failed: {e}")
    
    print("\n[OK] Provider comparison complete!")


if __name__ == "__main__":
    print("Log Analyzer - Usage Examples")
    print("="*70)
    
    # Uncomment the example you want to run:
    
    # example_openai_single_log()
    
    # example_mistral_single_log()
    
    # example_ollama_local()  # FREE - no API costs!
    
    # example_multiple_logs()
    
    # example_custom_configuration()
    
    # example_batch_processing()
    
    # example_compare_providers()
    
    print("\n💡 Tip: Edit this file to uncomment and run examples!")
    print("See log_analyzer.py for the main tool implementation.")

