#!/usr/bin/env python3
"""
Helper script to run specific logging demo examples
"""

import sys
import argparse


def run_all_examples():
    """Run all logging examples"""
    from logging_examples import main
    main()


def run_basic_examples():
    """Run basic examples (1-2)"""
    from logging_examples import example1_basic_usage, example2_component_logging
    
    print('\n=== Example 1: Basic Usage ===')
    example1_basic_usage()
    
    print('\n=== Example 2: Component Logging ===')
    example2_component_logging()


def run_advanced_examples():
    """Run advanced examples (5-8)"""
    from logging_examples import (
        example5_performance_timing,
        example6_exception_handling,
        example7_structured_logging,
        example8_log_levels
    )
    
    print('\n=== Example 5: Performance Timing ===')
    example5_performance_timing()
    
    print('\n=== Example 6: Exception Handling ===')
    example6_exception_handling()
    
    print('\n=== Example 7: Structured Logging ===')
    example7_structured_logging()
    
    print('\n=== Example 8: Log Levels ===')
    example8_log_levels()


def run_migration_examples():
    """Run migration examples"""
    from logging_demo_migration import main as migration_main
    migration_main()


def main():
    parser = argparse.ArgumentParser(description='Run logging framework demos')
    parser.add_argument(
        'demo_type',
        choices=['all', 'basic', 'advanced', 'migration'],
        default='all',
        help='Which demo to run'
    )
    
    args = parser.parse_args()
    
    if args.demo_type == 'all':
        run_all_examples()
    elif args.demo_type == 'basic':
        run_basic_examples()
    elif args.demo_type == 'advanced':
        run_advanced_examples()
    elif args.demo_type == 'migration':
        run_migration_examples()


if __name__ == '__main__':
    main()

