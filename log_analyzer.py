"""
Log Analyzer - Multi-LLM Failure Analysis Tool

This tool analyzes log files using various LLM providers (OpenAI, Mistral, etc.)
to identify failure reasons, error patterns, and root causes.

Supported LLM Providers:
- OpenAI (GPT-4, GPT-4o, GPT-3.5-turbo)
- Mistral AI (mistral-large, mistral-medium, mistral-small)
- Ollama (local models: llama2, mistral, etc.)
- Azure OpenAI
"""

import os
import sys
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import Counter

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Run: pip install pandas")
    sys.exit(1)

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_openai import ChatOpenAI
except ImportError:
    print("ERROR: LangChain not installed. Run: pip install langchain langchain-openai")
    sys.exit(1)

# Optional: Mistral support
try:
    from langchain_mistralai import ChatMistralAI
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

# Optional: Ollama support
try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class LogAnalyzer:
    """
    Analyzes log files using LLMs to identify failure reasons and patterns.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        verify_ssl: bool = True
    ):
        """
        Initialize the Log Analyzer
        
        Args:
            provider: LLM provider ("openai", "mistral", "ollama", "azure")
            model: Model name (provider-specific)
            api_key: API key for the provider
            base_url: Custom base URL (for Ollama or Azure)
            temperature: Model temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            verify_ssl: Whether to verify SSL certificates
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verify_ssl = verify_ssl
        
        # Set default models per provider
        if model is None:
            model = self._get_default_model(provider)
        
        self.model = model
        self.api_key = api_key or self._get_api_key(provider)
        self.base_url = base_url
        
        # Initialize the LLM
        self.llm = self._initialize_llm()
        
        print(f"[OK] Initialized Log Analyzer")
        print(f"  Provider: {self.provider}")
        print(f"  Model: {self.model}")
        print(f"  Temperature: {self.temperature}")
    
    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider"""
        defaults = {
            "openai": "gpt-4o-mini",
            "mistral": "mistral-large-latest",
            "ollama": "llama2",
            "azure": "gpt-4"
        }
        return defaults.get(provider, "gpt-4o-mini")
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment"""
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY"
        }
        env_var = env_vars.get(provider)
        if env_var:
            return os.getenv(env_var)
        return None
    
    def _initialize_llm(self):
        """Initialize the LLM based on provider"""
        if self.provider == "openai":
            return self._init_openai()
        elif self.provider == "mistral":
            return self._init_mistral()
        elif self.provider == "ollama":
            return self._init_ollama()
        elif self.provider == "azure":
            return self._init_azure()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _init_openai(self):
        """Initialize OpenAI LLM"""
        import httpx
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        http_client = None
        if not self.verify_ssl:
            http_client = httpx.Client(verify=False)
        
        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=self.api_key,
            http_client=http_client
        )
    
    def _init_mistral(self):
        """Initialize Mistral AI LLM"""
        if not MISTRAL_AVAILABLE:
            raise ImportError("Mistral support not installed. Run: pip install langchain-mistralai")
        
        if not self.api_key:
            raise ValueError("Mistral API key required. Set MISTRAL_API_KEY environment variable.")
        
        return ChatMistralAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            mistral_api_key=self.api_key
        )
    
    def _init_ollama(self):
        """Initialize Ollama (local LLM)"""
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama support not installed. Run: pip install langchain-community")
        
        base_url = self.base_url or "http://localhost:11434"
        
        return Ollama(
            model=self.model,
            base_url=base_url,
            temperature=self.temperature
        )
    
    def _init_azure(self):
        """Initialize Azure OpenAI LLM"""
        if not self.api_key:
            raise ValueError("Azure OpenAI API key required. Set AZURE_OPENAI_API_KEY.")
        
        if not self.base_url:
            raise ValueError("Azure base URL required. Provide base_url parameter.")
        
        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url
        )
    
    def parse_log_file(self, log_file_path: str) -> Dict[str, Any]:
        """
        Parse a log file and extract key information
        
        Args:
            log_file_path: Path to the log file
            
        Returns:
            Dictionary with parsed log data
        """
        print(f"\n[*] Parsing log file: {log_file_path}")
        
        if not os.path.exists(log_file_path):
            raise FileNotFoundError(f"Log file not found: {log_file_path}")
        
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()
        
        # Extract key patterns
        error_patterns = [
            r'ERROR[:\s]+(.*)',
            r'FATAL[:\s]+(.*)',
            r'CRITICAL[:\s]+(.*)',
            r'Exception[:\s]+(.*)',
            r'Traceback.*?(?=\n\n|\Z)',
            r'Failed[:\s]+(.*)',
            r'AssertionError[:\s]+(.*)',
        ]
        
        errors = []
        for pattern in error_patterns:
            matches = re.findall(pattern, log_content, re.IGNORECASE | re.DOTALL)
            errors.extend(matches)
        
        # Extract warnings
        warnings = re.findall(r'WARNING[:\s]+(.*)', log_content, re.IGNORECASE)
        
        # Extract stack traces
        stack_traces = re.findall(
            r'Traceback \(most recent call last\):.*?(?=\n\n|\Z)',
            log_content,
            re.DOTALL
        )
        
        # Count error types
        error_types = Counter()
        for error in errors:
            # Extract error type (first word or exception name)
            match = re.match(r'(\w+(?:Error|Exception))', error)
            if match:
                error_types[match.group(1)] += 1
        
        # Get file stats
        file_size = os.path.getsize(log_file_path)
        line_count = log_content.count('\n')
        
        parsed_data = {
            'file_path': log_file_path,
            'file_size': file_size,
            'line_count': line_count,
            'total_errors': len(errors),
            'total_warnings': len(warnings),
            'total_stack_traces': len(stack_traces),
            'error_types': dict(error_types.most_common(10)),
            'errors': errors[:50],  # First 50 errors
            'warnings': warnings[:20],  # First 20 warnings
            'stack_traces': stack_traces[:10],  # First 10 stack traces
            'log_snippet': log_content[-5000:]  # Last 5000 chars (usually most relevant)
        }
        
        print(f"[OK] Parsed log file")
        print(f"  Lines: {line_count:,}")
        print(f"  Size: {file_size:,} bytes")
        print(f"  Errors: {len(errors)}")
        print(f"  Warnings: {len(warnings)}")
        print(f"  Stack Traces: {len(stack_traces)}")
        
        return parsed_data
    
    def analyze_failure(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze log data using LLM to identify failure reasons
        
        Args:
            log_data: Parsed log data
            
        Returns:
            Analysis results
        """
        print(f"\n[*] Analyzing failures using {self.provider}/{self.model}...")
        
        # Create analysis prompt
        prompt = self._create_analysis_prompt(log_data)
        
        # Get LLM response
        try:
            messages = [
                SystemMessage(content="""You are an expert log analyst and debugging specialist.
Your task is to analyze log files and identify the root cause of failures.

Provide your analysis in the following structure:
1. Primary Failure Reason (1-2 sentences)
2. Error Type Classification (e.g., Configuration, Runtime, Network, etc.)
3. Key Error Messages (list top 3-5)
4. Root Cause Analysis (detailed explanation)
5. Recommended Actions (specific steps to fix)
6. Confidence Level (High/Medium/Low)

Be specific, actionable, and focus on the most critical issues."""),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Extract content based on response type
            if hasattr(response, 'content'):
                analysis_text = response.content
            else:
                analysis_text = str(response)
            
            # Parse the structured response
            analysis_result = self._parse_analysis_response(analysis_text, log_data)
            
            print(f"[OK] Analysis complete")
            print(f"  Primary Reason: {analysis_result.get('primary_reason', 'Unknown')[:80]}...")
            
            return analysis_result
            
        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            return {
                'error': str(e),
                'primary_reason': 'Analysis failed',
                'raw_log_data': log_data
            }
    
    def _create_analysis_prompt(self, log_data: Dict[str, Any]) -> str:
        """Create analysis prompt from log data"""
        
        prompt_parts = [
            "# Log Analysis Request\n",
            f"**File:** {log_data['file_path']}",
            f"**Lines:** {log_data['line_count']:,}",
            f"**Errors Found:** {log_data['total_errors']}",
            f"**Warnings Found:** {log_data['total_warnings']}",
            f"**Stack Traces:** {log_data['total_stack_traces']}\n"
        ]
        
        # Add error types
        if log_data['error_types']:
            prompt_parts.append("\n## Error Type Distribution:")
            for error_type, count in log_data['error_types'].items():
                prompt_parts.append(f"- {error_type}: {count} occurrences")
        
        # Add sample errors
        if log_data['errors']:
            prompt_parts.append("\n## Sample Error Messages:")
            for i, error in enumerate(log_data['errors'][:10], 1):
                error_snippet = error[:200].strip()
                prompt_parts.append(f"{i}. {error_snippet}")
        
        # Add stack traces
        if log_data['stack_traces']:
            prompt_parts.append("\n## Stack Traces:")
            for i, trace in enumerate(log_data['stack_traces'][:3], 1):
                trace_snippet = trace[:500].strip()
                prompt_parts.append(f"\n### Trace {i}:")
                prompt_parts.append(f"```\n{trace_snippet}\n```")
        
        # Add log snippet
        prompt_parts.append("\n## Recent Log Content:")
        prompt_parts.append(f"```\n{log_data['log_snippet']}\n```")
        
        prompt_parts.append("\n## Your Task:")
        prompt_parts.append("Analyze the above log data and provide:")
        prompt_parts.append("1. Primary failure reason")
        prompt_parts.append("2. Error classification")
        prompt_parts.append("3. Root cause analysis")
        prompt_parts.append("4. Recommended actions to fix the issue")
        
        return "\n".join(prompt_parts)
    
    def _parse_analysis_response(self, response_text: str, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse structured analysis response"""
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'provider': self.provider,
            'model': self.model,
            'file_analyzed': log_data['file_path'],
            'full_analysis': response_text,
            'log_stats': {
                'total_errors': log_data['total_errors'],
                'total_warnings': log_data['total_warnings'],
                'total_stack_traces': log_data['total_stack_traces'],
                'error_types': log_data['error_types']
            }
        }
        
        # Extract structured sections
        sections = {
            'primary_reason': r'(?:Primary Failure Reason|Failure Reason)[:\s]+(.*?)(?=\n\d\.|\n#|$)',
            'error_type': r'(?:Error Type Classification|Error Classification)[:\s]+(.*?)(?=\n\d\.|\n#|$)',
            'root_cause': r'(?:Root Cause Analysis|Root Cause)[:\s]+(.*?)(?=\n\d\.|\n#|$)',
            'recommendations': r'(?:Recommended Actions|Recommendations)[:\s]+(.*?)(?=\n\d\.|\n#|$)',
            'confidence': r'(?:Confidence Level|Confidence)[:\s]+(.*?)(?=\n|$)'
        }
        
        for key, pattern in sections.items():
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if match:
                result[key] = match.group(1).strip()
        
        return result
    
    def analyze_multiple_logs(
        self,
        log_directory: str,
        pattern: str = "*.log"
    ) -> Dict[str, Any]:
        """
        Analyze multiple log files in a directory
        
        Args:
            log_directory: Directory containing log files
            pattern: File pattern to match (e.g., "*.log", "error_*.txt")
            
        Returns:
            Combined analysis results
        """
        print(f"\n{'='*70}")
        print(f"Analyzing Multiple Log Files")
        print(f"{'='*70}")
        print(f"Directory: {log_directory}")
        print(f"Pattern: {pattern}\n")
        
        log_files = list(Path(log_directory).glob(pattern))
        
        if not log_files:
            raise FileNotFoundError(f"No log files found matching '{pattern}' in {log_directory}")
        
        print(f"[OK] Found {len(log_files)} log file(s)")
        
        results = {
            'summary': {
                'total_files': len(log_files),
                'analyzed_at': datetime.now().isoformat(),
                'provider': self.provider,
                'model': self.model
            },
            'individual_analyses': [],
            'common_patterns': None
        }
        
        # Analyze each log file
        for i, log_file in enumerate(log_files, 1):
            print(f"\n--- Analyzing {i}/{len(log_files)}: {log_file.name} ---")
            
            try:
                log_data = self.parse_log_file(str(log_file))
                analysis = self.analyze_failure(log_data)
                
                results['individual_analyses'].append({
                    'file': str(log_file),
                    'analysis': analysis
                })
                
            except Exception as e:
                print(f"[ERROR] Failed to analyze {log_file.name}: {e}")
                results['individual_analyses'].append({
                    'file': str(log_file),
                    'error': str(e)
                })
        
        # Find common patterns across all logs
        if len(results['individual_analyses']) > 1:
            results['common_patterns'] = self._find_common_patterns(results['individual_analyses'])
        
        return results
    
    def _find_common_patterns(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Find common failure patterns across multiple analyses"""
        
        error_types = []
        primary_reasons = []
        
        for analysis in analyses:
            if 'analysis' in analysis:
                if 'error_type' in analysis['analysis']:
                    error_types.append(analysis['analysis']['error_type'])
                if 'primary_reason' in analysis['analysis']:
                    primary_reasons.append(analysis['analysis']['primary_reason'])
        
        return {
            'most_common_error_types': Counter(error_types).most_common(5),
            'total_unique_failures': len(set(primary_reasons)),
            'failure_frequency': Counter(primary_reasons).most_common(10)
        }
    
    def save_analysis_report(
        self,
        analysis_results: Dict[str, Any],
        output_file: str = "log_analysis_report.md"
    ):
        """Save analysis results to a markdown report"""
        
        print(f"\n[*] Generating report...")
        
        report_lines = [
            "# Log Analysis Report\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Provider:** {self.provider}",
            f"**Model:** {self.model}\n",
            "---\n"
        ]
        
        # Check if it's a multi-file analysis
        if 'individual_analyses' in analysis_results:
            # Multiple files
            summary = analysis_results['summary']
            report_lines.extend([
                "## Summary\n",
                f"- Total Files Analyzed: {summary['total_files']}",
                f"- Analysis Date: {summary['analyzed_at']}\n"
            ])
            
            # Common patterns
            if analysis_results.get('common_patterns'):
                patterns = analysis_results['common_patterns']
                report_lines.extend([
                    "## Common Patterns Across All Logs\n",
                    f"- Unique Failure Types: {patterns['total_unique_failures']}",
                    "\n### Most Common Error Types:"
                ])
                
                for error_type, count in patterns['most_common_error_types']:
                    report_lines.append(f"- {error_type}: {count} occurrences")
                
                report_lines.append("")
            
            # Individual analyses
            report_lines.append("## Individual Log Analyses\n")
            
            for i, item in enumerate(analysis_results['individual_analyses'], 1):
                file_name = Path(item['file']).name
                report_lines.append(f"### {i}. {file_name}\n")
                
                if 'analysis' in item:
                    analysis = item['analysis']
                    report_lines.append(f"**File:** `{item['file']}`\n")
                    
                    if 'primary_reason' in analysis:
                        report_lines.append(f"**Primary Failure Reason:**\n{analysis['primary_reason']}\n")
                    
                    if 'error_type' in analysis:
                        report_lines.append(f"**Error Classification:** {analysis['error_type']}\n")
                    
                    if 'log_stats' in analysis:
                        stats = analysis['log_stats']
                        report_lines.extend([
                            "**Log Statistics:**",
                            f"- Errors: {stats['total_errors']}",
                            f"- Warnings: {stats['total_warnings']}",
                            f"- Stack Traces: {stats['total_stack_traces']}\n"
                        ])
                    
                    if 'full_analysis' in analysis:
                        report_lines.extend([
                            "**Detailed Analysis:**\n",
                            "```",
                            analysis['full_analysis'],
                            "```\n"
                        ])
                else:
                    report_lines.append(f"**Error:** {item.get('error', 'Unknown error')}\n")
                
                report_lines.append("---\n")
        
        else:
            # Single file analysis
            if 'file_analyzed' in analysis_results:
                report_lines.append(f"**File Analyzed:** `{analysis_results['file_analyzed']}`\n")
            
            if 'primary_reason' in analysis_results:
                report_lines.extend([
                    "## Primary Failure Reason\n",
                    f"{analysis_results['primary_reason']}\n"
                ])
            
            if 'error_type' in analysis_results:
                report_lines.extend([
                    "## Error Classification\n",
                    f"{analysis_results['error_type']}\n"
                ])
            
            if 'log_stats' in analysis_results:
                stats = analysis_results['log_stats']
                report_lines.extend([
                    "## Log Statistics\n",
                    f"- Total Errors: {stats['total_errors']}",
                    f"- Total Warnings: {stats['total_warnings']}",
                    f"- Stack Traces: {stats['total_stack_traces']}\n"
                ])
                
                if stats.get('error_types'):
                    report_lines.append("### Error Type Distribution:")
                    for error_type, count in stats['error_types'].items():
                        report_lines.append(f"- {error_type}: {count}")
                    report_lines.append("")
            
            if 'full_analysis' in analysis_results:
                report_lines.extend([
                    "## Detailed Analysis\n",
                    analysis_results['full_analysis'],
                    "\n"
                ])
        
        # Write report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"[OK] Report saved to: {output_file}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze log files using LLMs to identify failure reasons'
    )
    
    parser.add_argument(
        'log_path',
        help='Path to log file or directory containing log files'
    )
    parser.add_argument(
        '--provider',
        choices=['openai', 'mistral', 'ollama', 'azure'],
        default='openai',
        help='LLM provider to use'
    )
    parser.add_argument(
        '--model',
        help='Model name (provider-specific, uses defaults if not specified)'
    )
    parser.add_argument(
        '--api-key',
        help='API key for the provider'
    )
    parser.add_argument(
        '--base-url',
        help='Base URL (for Ollama or Azure)'
    )
    parser.add_argument(
        '--pattern',
        default='*.log',
        help='File pattern for directory analysis (default: *.log)'
    )
    parser.add_argument(
        '--output',
        default='log_analysis_report.md',
        help='Output report file (default: log_analysis_report.md)'
    )
    parser.add_argument(
        '--output-json',
        help='Also save results as JSON'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Model temperature (default: 0.3)'
    )
    parser.add_argument(
        '--no-ssl-verify',
        action='store_true',
        help='Disable SSL certificate verification'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    try:
        analyzer = LogAnalyzer(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            verify_ssl=not args.no_ssl_verify
        )
    except Exception as e:
        print(f"\nERROR: Failed to initialize analyzer: {e}")
        sys.exit(1)
    
    # Analyze log(s)
    try:
        log_path = Path(args.log_path)
        
        if log_path.is_dir():
            # Analyze directory
            results = analyzer.analyze_multiple_logs(
                str(log_path),
                pattern=args.pattern
            )
        elif log_path.is_file():
            # Analyze single file
            log_data = analyzer.parse_log_file(str(log_path))
            results = analyzer.analyze_failure(log_data)
        else:
            print(f"\nERROR: Path not found: {log_path}")
            sys.exit(1)
        
        # Save report
        analyzer.save_analysis_report(results, args.output)
        
        # Save JSON if requested
        if args.output_json:
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"[OK] JSON results saved to: {args.output_json}")
        
        print(f"\n{'='*70}")
        print("Analysis Complete!")
        print(f"{'='*70}")
        print(f"\nReport: {args.output}")
        if args.output_json:
            print(f"JSON: {args.output_json}")
        
    except Exception as e:
        print(f"\nERROR: Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

