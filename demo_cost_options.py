"""
Demo: Cost Control Options for Performance Analysis

This script demonstrates the different report generation options
and their cost implications.
"""
import os
from performance_analysis import PerformanceAnalyzer

def main():
    csv_file = r"C:\Users\rponnuru\Downloads\SubTestCountsMatrixView.csv"
    
    print("="*70)
    print("COST CONTROL DEMO")
    print("="*70)
    print("\nThis demo shows how to control report generation to manage costs.\n")
    
    # Option 1: Test Report Only (FREE)
    print("\n" + "="*70)
    print("OPTION 1: Test-Specific Report Only (FREE - $0.00)")
    print("="*70)
    print("✓ Detailed test metrics with pass/fail ratios")
    print("✓ Config-level breakdown")
    print("✓ Pattern analysis")
    print("✗ No AI-powered insights")
    print("\nGenerating...")
    
    analyzer1 = PerformanceAnalyzer(
        csv_file_path=csv_file,
        api_key="not-needed",  # No API key needed!
        model="gpt-4o-mini"
    )
    
    analyzer1.run_full_analysis(
        output_report="demo_report_1.md",
        output_raw="demo_data_1.json",
        generate_test_report=True,   # FREE
        generate_ai_report=False     # Skip to save costs
    )
    
    print("\n✓ Done! Cost: $0.00")
    print("Generated: demo_report_1_test_specific.md")
    
    # Option 2: Both Reports (Small Cost)
    print("\n" + "="*70)
    print("OPTION 2: Both Reports (~$0.003)")
    print("="*70)
    print("✓ Detailed test metrics")
    print("✓ AI-powered insights and recommendations")
    print("✓ Root cause hypothesis")
    print("✓ Pattern recognition")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n⚠ Skipping Option 2: OPENAI_API_KEY not set")
        print("To run this option, set your API key:")
        print('  $env:OPENAI_API_KEY="your-key-here"')
    else:
        print("\nGenerating...")
        
        analyzer2 = PerformanceAnalyzer(
            csv_file_path=csv_file,
            api_key=api_key,
            model="gpt-4o-mini",
            verify_ssl=False  # For corporate networks
        )
        
        report, stats = analyzer2.run_full_analysis(
            output_report="demo_report_2.md",
            output_raw="demo_data_2.json",
            generate_test_report=True,   # FREE
            generate_ai_report=True      # ~$0.003
        )
        
        print(f"\n✓ Done! Cost: ${stats['total_cost']:.4f}")
        print("Generated: demo_report_2.md + demo_report_2_test_specific.md")
    
    # Option 3: AI Report Only (Small Cost)
    print("\n" + "="*70)
    print("OPTION 3: AI Report Only (~$0.003)")
    print("="*70)
    print("✗ No detailed test-level breakdowns")
    print("✓ AI-powered executive summary")
    print("✓ High-level insights only")
    
    if not api_key:
        print("\n⚠ Skipping Option 3: OPENAI_API_KEY not set")
    else:
        print("\nGenerating...")
        
        analyzer3 = PerformanceAnalyzer(
            csv_file_path=csv_file,
            api_key=api_key,
            model="gpt-4o-mini",
            verify_ssl=False
        )
        
        report, stats = analyzer3.run_full_analysis(
            output_report="demo_report_3.md",
            output_raw="demo_data_3.json",
            generate_test_report=False,  # Skip details
            generate_ai_report=True      # ~$0.003
        )
        
        print(f"\n✓ Done! Cost: ${stats['total_cost']:.4f}")
        print("Generated: demo_report_3.md only")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n💡 Recommendations:")
    print("  • Daily monitoring: Use Option 1 (FREE)")
    print("  • Weekly deep-dive: Use Option 2 (both reports)")
    print("  • Executive summary: Use Option 3 (AI only)")
    print("\n📖 For more details, see COST_GUIDE.md")
    print("\n🎯 Command-line examples:")
    print("  python performance_analysis.py data.csv --no-ai-report     # FREE")
    print("  python performance_analysis.py data.csv                     # Both (~$0.003)")
    print("  python performance_analysis.py data.csv --report-type ai-only  # AI only")
    print("\n")

if __name__ == "__main__":
    main()

