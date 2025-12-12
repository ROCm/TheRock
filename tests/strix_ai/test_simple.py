"""
Simple test to verify pytest is working
"""

import pytest


def test_pytest_works():
    """Verify pytest is installed and working"""
    assert True
    print("✅ pytest is working!")


def test_environment():
    """Check Python environment"""
    import sys
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    assert sys.version_info >= (3, 8)


def test_automatic_workflow_trigger():
    """Test that workflows trigger on push"""
    import os
    amdgpu = os.getenv("AMDGPU_FAMILIES", "not-set")
    print(f"AMDGPU_FAMILIES: {amdgpu}")
    print("✅ Workflow triggered successfully!")
    assert True
