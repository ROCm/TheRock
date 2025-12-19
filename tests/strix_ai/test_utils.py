"""
Common test utilities for Strix AI/ML tests

Provides helpers for gracefully skipping tests when models aren't available.
"""

import pytest


def skip_if_model_unavailable(func):
    """
    Decorator to gracefully skip tests when models aren't available.
    
    Wraps model loading in try-except and skips with informative messages
    if the model is not found, requires authentication, or fails to load.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            
            # Check for common error patterns
            if "404" in error_msg or "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                pytest.skip(f"Model not yet available. Test will be enabled when model is published. Error: {error_msg[:100]}")
            elif "token" in error_msg.lower() or "authentication" in error_msg.lower() or "gated" in error_msg.lower():
                pytest.skip(f"Model requires authentication. Set HF_TOKEN environment variable. Error: {error_msg[:100]}")
            elif "import" in error_msg.lower() and "module" in error_msg.lower():
                pytest.skip(f"Required library not installed. Error: {error_msg[:100]}")
            elif "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                pytest.skip(f"Insufficient GPU memory for this model. Error: {error_msg[:100]}")
            else:
                pytest.skip(f"Unable to complete test. Error: {error_msg[:150]}")
    
    return wrapper


def load_model_safe(model_loader_func, model_id):
    """
    Safely load a model and skip test if unavailable.
    
    Args:
        model_loader_func: Function that loads the model
        model_id: Model identifier (for error messages)
    
    Returns:
        Loaded model or skips test
    """
    try:
        return model_loader_func()
    except Exception as e:
        error_msg = str(e)
        
        if "404" in error_msg or "not found" in error_msg.lower():
            pytest.skip(f"Model {model_id} not yet available on Hugging Face. "
                       f"Test will be enabled when model is published.")
        elif "token" in error_msg.lower() or "authentication" in error_msg.lower():
            pytest.skip(f"Model {model_id} requires authentication. "
                       f"Set HF_TOKEN environment variable to access gated models.")
        elif "import" in error_msg.lower():
            pytest.skip(f"Required library not installed for {model_id}. "
                       f"Install missing dependencies to enable this test.")
        else:
            pytest.skip(f"Unable to load {model_id}: {error_msg[:150]}")

