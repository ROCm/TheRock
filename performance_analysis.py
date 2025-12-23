"""
Performance Drop Analysis Tool using LangChain and Guardrails
Analyzes test performance data for config-specific and user-specific drops
"""

import pandas as pd
import json
import os
from typing import Dict, List, Any, Optional
from collections import Counter
import sys

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

# Guardrails imports
try:
    from guardrails import Guard
    from guardrails.hub import ToxicLanguage, CompetitorCheck, RestrictToTopic
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    print("WARNING: Guardrails AI not installed. Running without guardrails.")
    print("Install with: pip install guardrails-ai")


class PerformanceGuardrails:
    """Guardrails for performance analysis to ensure safe and relevant outputs"""
    
    def __init__(self):
        self.allowed_topics = [
            "performance analysis", "test infrastructure", "CI/CD systems",
            "hardware performance", "software testing", "configuration management",
            "resource allocation", "system optimization", "test execution",
            "performance metrics", "debugging", "troubleshooting"
        ]
        
        self.guard = None
        if GUARDRAILS_AVAILABLE:
            self._setup_guardrails()
    
    def _setup_guardrails(self):
        """Setup guardrails for output validation"""
        try:
            # Create guardrails to ensure:
            # 1. No toxic language
            # 2. Stay on topic (performance analysis)
            # 3. Professional and actionable output
            self.guard = Guard().use_many(
                ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail="exception"),
                RestrictToTopic(valid_topics=self.allowed_topics, disable_classifier=False, on_fail="reask")
            )
            print("[OK] Guardrails initialized successfully")
        except Exception as e:
            print(f"WARNING: Could not initialize all guardrails: {e}")
            self.guard = None
    
    def validate_output(self, text: str) -> tuple[bool, str, Optional[str]]:
        """
        Validate LLM output against guardrails
        
        Returns:
            tuple: (is_valid, validated_text, error_message)
        """
        if not GUARDRAILS_AVAILABLE or self.guard is None:
            return True, text, None
        
        try:
            validated_output = self.guard.validate(text)
            return True, validated_output.validated_output, None
        except Exception as e:
            return False, text, str(e)
    
    def validate_input(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate input data before processing
        
        Returns:
            tuple: (is_valid, error_message)
        """
        # Check for required fields
        required_fields = ['total_tests', 'total_configs']
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Check for reasonable data ranges
        if data['total_tests'] < 0 or data['total_configs'] < 0:
            return False, "Invalid negative values in data"
        
        if data['total_configs'] > 10000:
            return False, "Unreasonably large number of configurations"
        
        return True, None


class PerformanceAnalyzer:
    def __init__(self, csv_file_path: str, api_key: str = None, model: str = "gpt-4o", 
                 drop_zero_rows: bool = True, verify_ssl: bool = True):
        """
        Initialize the Performance Analyzer with LangChain
        
        Args:
            csv_file_path: Path to the CSV file containing test data
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env variable)
            model: Model to use (default: gpt-4o)
            drop_zero_rows: If True, drop rows with zero tests across all configs (default: True)
            verify_ssl: If False, disable SSL verification (for corporate networks) (default: True)
        """
        self.csv_file_path = csv_file_path
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model_name = model
        self.drop_zero_rows = drop_zero_rows
        self.verify_ssl = verify_ssl
        
        # Initialize LangChain components with SSL settings
        import httpx
        
        if not verify_ssl:
            print("[WARNING] SSL verification disabled - use only in trusted corporate networks")
            # Create custom httpx client with SSL verification disabled
            http_client = httpx.Client(verify=False)
        else:
            http_client = None  # Use default
        
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.7,
            max_tokens=4000,
            openai_api_key=self.api_key,
            http_client=http_client
        )
        
        # Initialize guardrails
        self.guardrails = PerformanceGuardrails()
        
        self.df = None
        self.config_columns = []
        self.metadata_columns = ['Features', 'Test_Category', 'Test_Name', 
                                'Software_Features', 'Hardware_Features', 
                                'Test_Area', 'Test_Execution_Mode', 
                                'Test_Plan_Name', 'Execution_Label']
        
    def load_data(self):
        """Load and parse the CSV file"""
        print(f"Loading data from {self.csv_file_path}...")
        self.df = pd.read_csv(self.csv_file_path)
        
        # Identify config columns (all columns except metadata)
        self.config_columns = [col for col in self.df.columns 
                              if col not in self.metadata_columns]
        
        initial_count = len(self.df)
        print(f"[OK] Loaded {initial_count} test cases across {len(self.config_columns)} configurations")
        
        # Optionally drop rows where ALL config columns have zero tests
        if self.drop_zero_rows:
            # Convert config columns to numeric, treating non-numeric as 0
            config_data = self.df[self.config_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Keep rows where at least one config has non-zero tests
            rows_with_data = (config_data != 0).any(axis=1)
            self.df = self.df[rows_with_data].reset_index(drop=True)
            
            dropped_count = initial_count - len(self.df)
            if dropped_count > 0:
                print(f"[OK] Dropped {dropped_count} test cases with zero tests across all configurations")
                print(f"[OK] Retained {len(self.df)} test cases with actual test data")
            else:
                print(f"[OK] All test cases have at least one non-zero configuration")
        
    def extract_config_details(self, config_name: str) -> Dict[str, str]:
        """Extract hardware, OS, user, and deployment type from config name"""
        parts = config_name.split('|')
        if len(parts) >= 5:
            return {
                'machine': parts[0].strip(),
                'os': parts[1].strip(),
                'hardware': parts[2].strip(),
                'gpu_count': parts[3].strip(),
                'user': parts[4].strip(),
                'deployment': parts[5].strip() if len(parts) > 5 else 'unknown'
            }
        return {}
    
    def analyze_performance_drops(self) -> Dict[str, Any]:
        """Analyze performance drops across configurations"""
        analysis_results = {
            'total_tests': len(self.df),
            'total_configs': len(self.config_columns),
            'zero_test_configs': [],
            'low_performance_configs': [],
            'config_comparison': [],
            'user_performance': {},
            'hardware_performance': {},
            'os_performance': {}
        }
        
        # Analyze each configuration
        for config in self.config_columns:
            config_details = self.extract_config_details(config)
            config_data = self.df[config]
            
            # Convert to numeric, handling any non-numeric values
            config_data = pd.to_numeric(config_data, errors='coerce').fillna(0)
            
            total_tests = config_data.sum()
            zero_tests = (config_data == 0).sum()
            avg_tests_per_suite = config_data.mean()
            
            # Track configs with zero or low performance
            if total_tests == 0:
                analysis_results['zero_test_configs'].append({
                    'config': config,
                    'details': config_details
                })
            elif avg_tests_per_suite < 1:
                analysis_results['low_performance_configs'].append({
                    'config': config,
                    'total_tests': int(total_tests),
                    'avg_per_suite': round(avg_tests_per_suite, 2),
                    'zero_count': int(zero_tests),
                    'details': config_details
                })
            
            # Aggregate by user
            if config_details.get('user'):
                user = config_details['user']
                if user not in analysis_results['user_performance']:
                    analysis_results['user_performance'][user] = {
                        'total_tests': 0,
                        'configs': 0,
                        'avg_per_config': 0
                    }
                analysis_results['user_performance'][user]['total_tests'] += int(total_tests)
                analysis_results['user_performance'][user]['configs'] += 1
            
            # Aggregate by hardware
            if config_details.get('hardware'):
                hw = config_details['hardware']
                if hw not in analysis_results['hardware_performance']:
                    analysis_results['hardware_performance'][hw] = {
                        'total_tests': 0,
                        'configs': 0
                    }
                analysis_results['hardware_performance'][hw]['total_tests'] += int(total_tests)
                analysis_results['hardware_performance'][hw]['configs'] += 1
            
            # Aggregate by OS
            if config_details.get('os'):
                os_name = config_details['os']
                if os_name not in analysis_results['os_performance']:
                    analysis_results['os_performance'][os_name] = {
                        'total_tests': 0,
                        'configs': 0
                    }
                analysis_results['os_performance'][os_name]['total_tests'] += int(total_tests)
                analysis_results['os_performance'][os_name]['configs'] += 1
        
        # Calculate averages for user performance
        for user, data in analysis_results['user_performance'].items():
            data['avg_per_config'] = round(data['total_tests'] / data['configs'], 2)
        
        return analysis_results
    
    def identify_test_failures(self) -> List[Dict[str, Any]]:
        """
        Identify tests that failed across multiple configurations.
        Only considers configs where tests actually executed (non-zero).
        Captures specific config names where tests failed.
        """
        test_failures = []
        
        for idx, row in self.df.iterrows():
            test_info = {
                'test_category': row['Test_Category'],
                'test_name': row['Test_Name'],
                'features': row['Features']
            }
            
            # Get numeric data for all configs
            numeric_data = pd.to_numeric(row[self.config_columns], errors='coerce').fillna(0)
            
            # Get lists of configs by status
            executed_configs_list = [col for col in self.config_columns if numeric_data[col] > 0]
            failed_configs_list = [col for col in self.config_columns if numeric_data[col] == 0]
            
            # Count configs where test executed (non-zero)
            executed_configs = len(executed_configs_list)
            
            # Skip tests that never executed anywhere
            if executed_configs == 0:
                continue
            
            # Count configs where test had zero results
            zero_configs = len(failed_configs_list)
            
            # Calculate failure rate based on configs where test could have run
            failure_rate = (zero_configs / len(self.config_columns)) * 100
            
            # Only include tests with significant failure rates
            if failure_rate > 50:  # More than 50% of configs have zero results
                test_failures.append({
                    **test_info,
                    'executed_on_configs': int(executed_configs),
                    'failed_on_configs': int(zero_configs),
                    'total_configs': len(self.config_columns),
                    'failure_rate': round(failure_rate, 2),
                    'success_rate': round((executed_configs / len(self.config_columns)) * 100, 2),
                    'configs_with_executions': executed_configs_list[:10],  # Limit to first 10 for readability
                    'configs_with_failures': failed_configs_list[:10]  # Limit to first 10 for readability
                })
        
        return sorted(test_failures, key=lambda x: x['failure_rate'], reverse=True)
    
    def create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create LangChain prompt template for performance analysis"""
        
        # Define the system message
        system_template = """You are an expert performance analyst specializing in test infrastructure and CI/CD systems. 
You excel at identifying patterns in test data, diagnosing performance issues, and providing actionable recommendations.

Your analysis should be:
1. Professional and objective
2. Data-driven with specific metrics
3. Actionable with clear recommendations
4. Focused on performance testing and infrastructure
5. Free from speculation - only analyze what the data shows

Format your response as a detailed markdown report with clear sections and bullet points."""

        # Define the human message template
        human_template = """
Analyze the following performance test data and identify performance drops and issues.

## Test Data Summary:
- Total Test Suites: {total_tests}
- Total Configurations Analyzed: {total_configs}
- Configurations with LOW performance: {low_configs_count}

## Configuration-Specific Performance Issues:

### Configs with Low Performance (Focus Area):
{low_performance_configs}

## User-Specific Performance Analysis:

Top Users by Average Tests per Config:
{top_users}

Bottom Users by Average Tests per Config:
{bottom_users}

## Hardware-Specific Performance Analysis:

Top Hardware Platforms by Total Tests:
{top_hardware}

Bottom Hardware Platforms by Total Tests:
{bottom_hardware}

## Test Performance Issues Across Configurations:

Tests with highest failure rates (>50% configs have zero tests).
For each test, you'll see:
- Which specific configs successfully executed the test
- Which specific configs failed to execute the test
- Success rate and failure rate percentages

{test_failures}

## Analysis Requirements:

Focus your analysis on configs where tests ARE running, not on configs with no tests.
Calculate failure rates based on configs where tests were actually executed.

Please provide a comprehensive report with the following sections:

1. **Executive Summary**: 
   - Overall health of active test infrastructure
   - Key performance concerns in running configurations

2. **Configuration-Specific Issues**:
   - Identify configs with critical performance drops (where tests run but with low counts)
   - Common patterns in underperforming configurations
   - Potential root causes (hardware, OS, deployment type)

3. **User-Specific Performance Analysis**:
   - Users with consistently low test execution rates
   - Potential capacity or resource issues
   - Recommendations for workload distribution

4. **Hardware/Platform Issues**:
   - Hardware platforms with poor performance (where tests run)
   - OS-specific issues affecting test execution
   - GPU configuration problems

5. **Test-Specific Failures**:
   - Tests with low success rates across configs where they execute
   - Identify patterns in configs_with_failures (common OS, hardware, users)
   - Tests with platform-specific compatibility issues
   - Use the configs_with_executions and configs_with_failures lists to identify patterns
   - Note: Focus on execution_rate and success_rate, not just presence/absence

6. **Actionable Recommendations**:
   - Immediate actions to improve performance on underperforming configs
   - Long-term improvements for test infrastructure
   - Resource allocation and optimization suggestions

Please format your response as a detailed markdown report.
"""

        # Create the prompt template
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        
        return chat_prompt
    
    def get_ai_analysis(self, analysis_data: Dict[str, Any], 
                       test_failures: List[Dict[str, Any]]) -> tuple[str, Dict[str, Any]]:
        """
        Get analysis from LLM using LangChain
        
        Returns:
            tuple: (analysis_report, usage_stats)
        """
        print("\n" + "="*70)
        print("Sending data to LLM for analysis...")
        print(f"Model: {self.model_name}")
        print("="*70)
        
        # Validate input data with guardrails
        is_valid, error_msg = self.guardrails.validate_input(analysis_data)
        if not is_valid:
            raise ValueError(f"Input validation failed: {error_msg}")
        
        # Sort user performance by avg_per_config
        sorted_users = sorted(
            analysis_data['user_performance'].items(),
            key=lambda x: x[1]['avg_per_config'],
            reverse=True
        )
        
        # Sort hardware performance
        sorted_hardware = sorted(
            analysis_data['hardware_performance'].items(),
            key=lambda x: x[1]['total_tests'],
            reverse=True
        )
        
        # Prepare input variables for the chain
        # Focus on active configs and performance issues, not missing configs
        input_vars = {
            'total_tests': analysis_data['total_tests'],
            'total_configs': analysis_data['total_configs'],
            'low_configs_count': len(analysis_data['low_performance_configs']),
            'low_performance_configs': json.dumps(analysis_data['low_performance_configs'][:20], indent=2),
            'top_users': json.dumps(dict(sorted_users[:10]), indent=2),
            'bottom_users': json.dumps(dict(sorted_users[-10:]), indent=2),
            'top_hardware': json.dumps(dict(sorted_hardware[:15]), indent=2),
            'bottom_hardware': json.dumps(dict(sorted_hardware[-15:]), indent=2),
            'test_failures': json.dumps(test_failures[:20], indent=2)
        }
        
        try:
            # Create the analysis prompt
            prompt = self.create_analysis_prompt()
            
            # Create the chain using LCEL (modern LangChain approach)
            chain = prompt | self.llm | StrOutputParser()
            
            # Run the chain with callback to track usage
            with get_openai_callback() as cb:
                report = chain.invoke(input_vars)
                
                usage_stats = {
                    'total_tokens': cb.total_tokens,
                    'prompt_tokens': cb.prompt_tokens,
                    'completion_tokens': cb.completion_tokens,
                    'total_cost': cb.total_cost
                }
                
                print(f"\n[OK] Analysis complete")
                print(f"  - Tokens used: {cb.total_tokens}")
                print(f"  - Estimated cost: ${cb.total_cost:.4f}")
            
            # Validate output with guardrails
            is_valid, validated_report, error_msg = self.guardrails.validate_output(report)
            
            if not is_valid:
                print(f"\nWARNING: Guardrail validation failed: {error_msg}")
                print("Proceeding with original output but please review carefully.")
            else:
                report = validated_report
                print("[OK] Output validated by guardrails")
            
            return report, usage_stats
            
        except Exception as e:
            print(f"Error during LLM analysis: {e}")
            raise
    
    def save_report(self, report: str, usage_stats: Dict[str, Any], 
                   output_file: str = "performance_report.md"):
        """Save the analysis report to a file"""
        
        # Add metadata header to report
        metadata = f"""# Performance Analysis Report

**Generated using:**
- Framework: LangChain
- Model: {self.model_name}
- Guardrails: {"Enabled" if GUARDRAILS_AVAILABLE else "Disabled"}

**Usage Statistics:**
- Total Tokens: {usage_stats['total_tokens']}
- Prompt Tokens: {usage_stats['prompt_tokens']}
- Completion Tokens: {usage_stats['completion_tokens']}
- Estimated Cost: ${usage_stats['total_cost']:.4f}

---

"""
        
        full_report = metadata + report
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_report)
        print(f"\n[OK] Report saved to: {output_file}")
    
    def save_raw_analysis(self, analysis_data: Dict[str, Any], 
                         test_failures: List[Dict[str, Any]],
                         output_file: str = "raw_analysis.json"):
        """Save raw analysis data to JSON"""
        combined_data = {
            'analysis': analysis_data,
            'test_failures': test_failures,
            'metadata': {
                'model': self.model_name,
                'framework': 'LangChain',
                'guardrails_enabled': GUARDRAILS_AVAILABLE
            }
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2)
        print(f"[OK] Raw analysis data saved to: {output_file}")
    
    def generate_test_specific_report(self, test_failures: List[Dict[str, Any]], 
                                      output_file: str = "test_specific_report.md"):
        """
        Generate a detailed test-specific report showing pass/fail ratios and drops
        
        Args:
            test_failures: List of test failure data
            output_file: Output markdown file name
        """
        print("\nGenerating test-specific detailed report...")
        
        report_lines = [
            "# Test-Specific Performance Report",
            "",
            f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Tests Analyzed**: {len(test_failures)}",
            "",
            "This report provides detailed pass/fail analysis for each test case.",
            "",
            "---",
            ""
        ]
        
        # Sort tests by failure rate (worst first)
        sorted_tests = sorted(test_failures, key=lambda x: x['failure_rate'], reverse=True)
        
        for idx, test in enumerate(sorted_tests, 1):
            # Extract configuration details from failed/successful configs
            failed_configs = test.get('configs_with_failures', [])
            success_configs = test.get('configs_with_executions', [])
            
            report_lines.extend([
                f"## {idx}. {test['test_name']}",
                "",
                f"**Category**: {test['test_category']}",
                f"**Feature**: {test['features']}",
                "",
                "### Performance Summary",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| **Pass Ratio** | {test['success_rate']:.2f}% ({test['executed_on_configs']}/{test['total_configs']} configs) |",
                f"| **Fail Ratio** | {test['failure_rate']:.2f}% ({test['failed_on_configs']}/{test['total_configs']} configs) |",
                f"| **Performance Drop** | {test['failure_rate']:.2f}% of configs not executing |",
                "",
                f"### Execution Analysis",
                "",
                f"- **Successfully Executed On**: {test['executed_on_configs']} configurations",
                f"- **Failed/Not Executed On**: {test['failed_on_configs']} configurations",
                f"- **Total Configurations**: {test['total_configs']}",
                "",
            ])
            
            # Successful configs
            if success_configs:
                report_lines.extend([
                    "### ✓ Configs Where Test PASSED",
                    "",
                    "| # | Configuration | OS | Hardware | User |",
                    "|---|---------------|----|-----------|----|"
                ])
                
                for i, config in enumerate(success_configs[:10], 1):
                    details = self.extract_config_details(config)
                    os_name = details.get('os', 'N/A')
                    hardware = details.get('hardware', 'N/A')
                    user = details.get('user', 'N/A')
                    report_lines.append(f"| {i} | {config[:50]}... | {os_name} | {hardware} | {user} |")
                
                if len(success_configs) > 10:
                    report_lines.append(f"\n*...and {len(success_configs) - 10} more configs*")
                report_lines.append("")
            
            # Failed configs
            if failed_configs:
                report_lines.extend([
                    "### ✗ Configs Where Test FAILED/NOT EXECUTED",
                    "",
                    "| # | Configuration | OS | Hardware | User |",
                    "|---|---------------|----|-----------|----|"
                ])
                
                for i, config in enumerate(failed_configs[:10], 1):
                    details = self.extract_config_details(config)
                    os_name = details.get('os', 'N/A')
                    hardware = details.get('hardware', 'N/A')
                    user = details.get('user', 'N/A')
                    report_lines.append(f"| {i} | {config[:50]}... | {os_name} | {hardware} | {user} |")
                
                if len(failed_configs) > 10:
                    report_lines.append(f"\n*...and {len(failed_configs) - 10} more configs*")
                report_lines.append("")
            
            # Pattern analysis
            report_lines.extend([
                "### Pattern Analysis",
                "",
            ])
            
            # Analyze OS patterns
            failed_os = [self.extract_config_details(c).get('os', 'Unknown') for c in failed_configs[:10]]
            success_os = [self.extract_config_details(c).get('os', 'Unknown') for c in success_configs[:10]]
            
            failed_os_count = Counter(failed_os)
            success_os_count = Counter(success_os)
            
            report_lines.extend([
                "**Most Common OS in Failures:**",
                ""
            ])
            for os_name, count in failed_os_count.most_common(3):
                report_lines.append(f"- {os_name}: {count} configs")
            
            report_lines.extend([
                "",
                "**Most Common OS in Successes:**",
                ""
            ])
            for os_name, count in success_os_count.most_common(3):
                report_lines.append(f"- {os_name}: {count} configs")
            
            # Analyze hardware patterns
            failed_hw = [self.extract_config_details(c).get('hardware', 'Unknown') for c in failed_configs[:10]]
            success_hw = [self.extract_config_details(c).get('hardware', 'Unknown') for c in success_configs[:10]]
            
            failed_hw_count = Counter(failed_hw)
            success_hw_count = Counter(success_hw)
            
            report_lines.extend([
                "",
                "**Most Common Hardware in Failures:**",
                ""
            ])
            for hw, count in failed_hw_count.most_common(3):
                report_lines.append(f"- {hw}: {count} configs")
            
            report_lines.extend([
                "",
                "**Most Common Hardware in Successes:**",
                ""
            ])
            for hw, count in success_hw_count.most_common(3):
                report_lines.append(f"- {hw}: {count} configs")
            
            report_lines.extend([
                "",
                "---",
                ""
            ])
        
        # Write report
        report_content = "\n".join(report_lines)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"[OK] Test-specific report saved to: {output_file}")
        return output_file
    
    def run_full_analysis(self, output_report: str = "performance_report.md",
                         output_raw: str = "raw_analysis.json",
                         generate_test_report: bool = True):
        """Run the complete analysis pipeline"""
        print("="*70)
        print("Performance Drop Analysis Tool")
        print("Powered by LangChain + Guardrails")
        print("="*70)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Analyze performance drops
        print("\nAnalyzing performance drops...")
        analysis_data = self.analyze_performance_drops()
        print(f"[OK] Analyzing {len(analysis_data['low_performance_configs'])} configs with low performance")
        
        # Step 3: Identify test failures  
        print("\nIdentifying test performance issues...")
        test_failures = self.identify_test_failures()
        print(f"[OK] Found {len(test_failures)} tests with low success rates across configs")
        
        # Step 4: Save raw analysis
        self.save_raw_analysis(analysis_data, test_failures, output_raw)
        
        # Step 5: Generate test-specific detailed report
        if generate_test_report:
            test_report_file = output_report.replace('.md', '_test_specific.md')
            self.generate_test_specific_report(test_failures, test_report_file)
        
        # Step 6: Get AI analysis using LangChain
        ai_report, usage_stats = self.get_ai_analysis(analysis_data, test_failures)
        
        # Step 7: Save final report
        self.save_report(ai_report, usage_stats, output_report)
        
        print("\n" + "="*70)
        print("Analysis Complete!")
        print("="*70)
        print(f"\nGenerated files:")
        print(f"  1. {output_report} - AI-generated analysis report")
        print(f"  2. {output_raw} - Raw analysis data (JSON)")
        if generate_test_report:
            print(f"  3. {test_report_file} - Test-specific detailed report")
        
        return ai_report, usage_stats


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze test performance data using LangChain and Guardrails'
    )
    parser.add_argument(
        'csv_file',
        help='Path to the CSV file containing test data'
    )
    parser.add_argument(
        '--api-key',
        help='OpenAI API key (or set OPENAI_API_KEY environment variable)',
        default=None
    )
    parser.add_argument(
        '--output-report',
        help='Output file for the analysis report',
        default='performance_report.md'
    )
    parser.add_argument(
        '--output-raw',
        help='Output file for raw analysis data (JSON)',
        default='raw_analysis.json'
    )
    parser.add_argument(
        '--model',
        help='OpenAI model to use',
        default='gpt-4o',
        choices=['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo']
    )
    parser.add_argument(
        '--keep-zero-rows',
        help='Keep test rows that have zero tests across all configurations',
        action='store_true',
        default=False
    )
    
    args = parser.parse_args()
    
    # Check if API key is available
    if not args.api_key and not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OpenAI API key not provided!")
        print("Please either:")
        print("  1. Set the OPENAI_API_KEY environment variable")
        print("  2. Use the --api-key argument")
        sys.exit(1)
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"ERROR: CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    # Run analysis
    try:
        analyzer = PerformanceAnalyzer(
            csv_file_path=args.csv_file,
            api_key=args.api_key,
            model=args.model,
            drop_zero_rows=not args.keep_zero_rows  # Invert the flag
        )
        analyzer.run_full_analysis(
            output_report=args.output_report,
            output_raw=args.output_raw
        )
    except Exception as e:
        print(f"\nERROR: Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
