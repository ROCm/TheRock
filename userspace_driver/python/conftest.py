"""Exclude standalone test scripts from pytest collection.

test_hw_hello.py and test_debug_discovery.py are standalone scripts
meant to be run directly (python test_hw_hello.py), not via pytest.
"""

collect_ignore = ["test_hw_hello.py", "test_debug_discovery.py"]
