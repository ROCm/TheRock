"""
Example usage of the Performance Analysis Tool
This script demonstrates how to use the tool programmatically
"""

import os
from performance_analysis import PerformanceAnalyzer

def example_basic_usage():
    """Basic usage example"""
    print("Example 1: Basic Usage")
    print("-" * 50)
    
    # Path to your CSV file
    csv_file = r"C:\Users\rponnuru\Downloads\SubTestCountsMatrixView.csv"
    
    # API key - REPLACE WITH YOUR ACTUAL KEY or use environment variable
    api_key = os.getenv('OPENAI_API_KEY') or "your-api-key-here"
    
    if api_key == "your-api-key-here":
        print("ERROR: Please set your OPENAI_API_KEY")
        print("Either set environment variable or edit this file with your key")
        return
    
    # Create analyzer instance with the API key
    analyzer = PerformanceAnalyzer(
        csv_file_path=csv_file,
        api_key=api_key,  # Pass the API key directly
        model="gpt-4o-mini"  # Using cheaper model for testing
    )
    
    # Run full analysis
    report, usage_stats = analyzer.run_full_analysis(
        output_report="performance_report.md",
        output_raw="raw_analysis.json"
    )
    
    print(f"\nAnalysis complete!")
    print(f"Tokens used: {usage_stats['total_tokens']}")
    print(f"Cost: ${usage_stats['total_cost']:.4f}")


def example_with_custom_api_key():
    """Example with custom API key"""
    print("\nExample 2: With Custom API Key")
    print("-" * 50)
    
    csv_file = r"C:\Users\rponnuru\Downloads\SubTestCountsMatrixView.csv"
    api_key = "your-api-key-here"  # Replace with your key
    
    analyzer = PerformanceAnalyzer(
        csv_file_path=csv_file,
        api_key=api_key,
        model="gpt-4o-mini"  # Using cheaper model
    )
    
    report, usage_stats = analyzer.run_full_analysis()
    print("Done!")


def example_step_by_step():
    """Example running analysis step by step"""
    print("\nExample 3: Step-by-Step Analysis")
    print("-" * 50)
    
    csv_file = r"C:\Users\rponnuru\Downloads\SubTestCountsMatrixView.csv"
    
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        return
    
    # Create analyzer
    analyzer = PerformanceAnalyzer(csv_file_path=csv_file)
    
    # Step 1: Load data
    analyzer.load_data()
    print(f"Loaded {len(analyzer.df)} test cases")
    
    # Step 2: Analyze performance
    analysis_data = analyzer.analyze_performance_drops()
    print(f"Found {len(analysis_data['zero_test_configs'])} configs with zero tests")
    print(f"Found {len(analysis_data['low_performance_configs'])} low performance configs")
    
    # Step 3: Identify test failures
    test_failures = analyzer.identify_test_failures()
    print(f"Found {len(test_failures)} tests with high failure rates")
    
    # Step 4: Get AI analysis
    report, usage_stats = analyzer.get_ai_analysis(analysis_data, test_failures)
    
    # Step 5: Save reports
    analyzer.save_report(report, usage_stats, "my_custom_report.md")
    analyzer.save_raw_analysis(analysis_data, test_failures, "my_custom_data.json")
    
    print("Step-by-step analysis complete!")


def example_data_exploration():
    """Example of exploring the data without AI analysis"""
    print("\nExample 4: Data Exploration Only")
    print("-" * 50)
    
    csv_file = r"C:\Users\rponnuru\Downloads\SubTestCountsMatrixView.csv"
    
    # No API key needed for data exploration
    # By default, rows with all zeros are dropped for cleaner analysis
    analyzer = PerformanceAnalyzer(csv_file_path=csv_file, drop_zero_rows=True)
    
    # Load and analyze data
    analyzer.load_data()
    analysis_data = analyzer.analyze_performance_drops()
    test_failures = analyzer.identify_test_failures()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total test suites: {analysis_data['total_tests']}")
    print(f"Total configurations: {analysis_data['total_configs']}")
    print(f"Configs with zero tests: {len(analysis_data['zero_test_configs'])}")
    print(f"Configs with low performance: {len(analysis_data['low_performance_configs'])}")
    
    print("\n=== Top 5 Users by Performance ===")
    sorted_users = sorted(
        analysis_data['user_performance'].items(),
        key=lambda x: x[1]['avg_per_config'],
        reverse=True
    )
    for user, stats in sorted_users[:5]:
        print(f"  {user}: {stats['avg_per_config']:.2f} avg tests/config")
    
    print("\n=== Top 5 Hardware Platforms ===")
    sorted_hw = sorted(
        analysis_data['hardware_performance'].items(),
        key=lambda x: x[1]['total_tests'],
        reverse=True
    )
    for hw, stats in sorted_hw[:5]:
        print(f"  {hw}: {stats['total_tests']} total tests")
    
    print("\n=== Tests with Highest Failure Rates ===")
    for test in test_failures[:5]:
        print(f"  {test['test_name']}: {test['failure_rate']:.1f}% failure rate")
    
    # Save just the raw data
    analyzer.save_raw_analysis(analysis_data, test_failures, "exploration_data.json")
    print("\nData exploration complete! Check exploration_data.json")


def example_keep_zero_rows():
    """Example keeping rows with all zeros"""
    print("\nExample 5: Keep All Rows Including Zeros")
    print("-" * 50)
    
    csv_file = r"C:\Users\rponnuru\Downloads\SubTestCountsMatrixView.csv"
    
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        return
    
    # Keep all rows, even those with zero tests across all configs
    analyzer = PerformanceAnalyzer(
        csv_file_path=csv_file,
        model="gpt-4o-mini",
        drop_zero_rows=False  # Keep zero rows
    )
    
    report, usage_stats = analyzer.run_full_analysis(
        output_report="report_with_zeros.md",
        output_raw="data_with_zeros.json"
    )
    
    print("Analysis complete with all rows included!")


if __name__ == "__main__":
    print("Performance Analysis Tool - Usage Examples")
    print("=" * 70)
    
    # Uncomment the example you want to run:
    
    example_basic_usage()
    
    # example_with_custom_api_key()
    
    # example_step_by_step()
    
    # example_data_exploration()  # No API key needed
    
    # example_keep_zero_rows()  # Include rows with all zeros

