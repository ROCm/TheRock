"""
ROCm Component Test Kit

A comprehensive testing framework for ROCm components on MI300/MI350 hardware.
"""

__version__ = "1.0.0"
__author__ = "ROCm Team"

from .hardware_detector import detect_hardware, check_compatibility, HardwareInfo
from .test_runner import TestRunner, TestResult

__all__ = [
    'detect_hardware',
    'check_compatibility',
    'HardwareInfo',
    'TestRunner',
    'TestResult',
]
