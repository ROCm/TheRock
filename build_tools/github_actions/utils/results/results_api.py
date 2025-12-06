"""API client for submitting test results."""

import json
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..logger import log
from ..exceptions import (
    APIAuthenticationError,
    APIValidationError,
    APIServerError
)
from ..system import (
    format_memory_size,
    format_cache_size,
    format_clock_speed
)


class ResultsAPI:
    """API client for submitting test results with authentication and fallback support."""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None, fallback_url: Optional[str] = None):
        """Initialize API client.
        
        Args:
            api_url: Base URL for the primary API
            api_key: Optional API key for authentication
            fallback_url: Optional fallback URL if primary fails
        """
        self.api_url = api_url.rstrip('/')
        self.fallback_url = fallback_url.rstrip('/') if fallback_url else None
        self.api_key = api_key
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}'
            })
        
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def submit_results(self, payload: Dict[str, Any]) -> bool:
        """Submit test results to API with fallback support.
        
        Args:
            payload: Results payload dictionary
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            APIAuthenticationError: Authentication failures
            APIValidationError: Invalid payload
            APIServerError: Server-side errors
        """
        # Try primary URL first
        primary_success = self._try_submit(self.api_url, payload, is_fallback=False)
        if primary_success:
            return True
        
        # Try fallback URL if configured
        if self.fallback_url:
            log.warning(f"Primary API failed, trying fallback: {self.fallback_url}")
            return self._try_submit(self.fallback_url, payload, is_fallback=True)
        
        return False
    
    def _try_submit(self, base_url: str, payload: Dict[str, Any], is_fallback: bool = False) -> bool:
        """Try to submit results to a specific URL.
        
        Args:
            base_url: Base URL to submit to
            payload: Results payload dictionary
            is_fallback: Whether this is a fallback attempt
            
        Returns:
            True if successful, False otherwise
        """
        try:
            endpoint = f"{base_url}/api/v1/rock-ci-results"
            url_type = "fallback" if is_fallback else "primary"
            
            log.debug(f"Payload size: {len(json.dumps(payload))} bytes")
            
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                log.info(f"✓ Results submitted successfully to {url_type} API")
                try:
                    response_data = response.json()
                    log.debug(f"API Response: {json.dumps(response_data, indent=2)}")
                except Exception:
                    log.debug(f"Response text: {response.text[:500]}")
                return True
            elif response.status_code == 201:
                log.info(f"✓ Results created successfully on {url_type} API")
                try:
                    response_data = response.json()
                    log.debug(f"API Response: {json.dumps(response_data, indent=2)}")
                except Exception:
                    log.debug(f"Response text: {response.text[:500]}")
                return True
            else:
                self._handle_http_error(response)
                return False
                
        except requests.exceptions.Timeout as e:
            url_type = "fallback" if is_fallback else "primary"
            log.warning(f"✗ {url_type.capitalize()} API Request Timed Out: {e}")
            if not is_fallback:
                log.debug("  Will try fallback URL if configured")
            return False
            
        except requests.exceptions.ConnectionError as e:
            url_type = "fallback" if is_fallback else "primary"
            log.warning(f"✗ {url_type.capitalize()} API Connection Failed: {e}")
            if not is_fallback:
                log.debug("  Will try fallback URL if configured")
            return False
            
        except requests.exceptions.HTTPError as e:
            url_type = "fallback" if is_fallback else "primary"
            log.warning(f"✗ {url_type.capitalize()} API HTTP Error: {e}")
            return False
            
        except json.JSONDecodeError as e:
            url_type = "fallback" if is_fallback else "primary"
            log.warning(f"✗ {url_type.capitalize()} API Invalid JSON Response: {e}")
            return False
            
        except Exception as e:
            url_type = "fallback" if is_fallback else "primary"
            log.warning(f"✗ {url_type.capitalize()} API Unexpected Error: {e}")
            return False
    
    def _handle_http_error(self, response: requests.Response):
        """Handle HTTP error responses with specific exceptions.
        
        Args:
            response: HTTP response object
            
        Raises:
            APIAuthenticationError: For 401 errors
            APIValidationError: For 400 errors
            APIServerError: For 500 errors
        """
        status_code = response.status_code
        
        if status_code == 401:
            log.error("✗ API Authentication Failed")
            log.error("  Solutions:")
            log.error("    1. Check API key in config (ResultsAPI.APIKey)")
            log.error("    2. Verify API key is valid and not expired")
            log.error("    3. Contact API administrator for new key")
            raise APIAuthenticationError(f"Authentication failed (401): Invalid or missing API key")
            
        elif status_code == 400:
            log.error("✗ Invalid Payload")
            log.error(f"  Response: {response.text[:500]}")
            log.error("  Solutions:")
            log.error("    1. Check payload structure matches API schema")
            log.error("    2. Verify all required fields are present")
            log.error("    3. Check field data types are correct")
            log.error("    4. Review API documentation")
            raise APIValidationError(f"Invalid payload (400): {response.text[:200]}")
            
        elif status_code == 403:
            log.error("✗ API Access Forbidden")
            log.error("  Solutions:")
            log.error("    1. Verify API key has required permissions")
            log.error("    2. Contact API administrator")
            raise APIAuthenticationError(f"Access forbidden (403): Insufficient permissions")
            
        elif status_code == 404:
            log.error("✗ API Endpoint Not Found")
            log.error(f"  URL: {response.url}")
            log.error("  Solutions:")
            log.error("    1. Verify API URL in config is correct")
            log.error("    2. Check API version compatibility")
            log.error("    3. Contact API administrator")
            raise APIServerError(f"Endpoint not found (404): {response.url}")
            
        elif status_code == 500:
            log.error("✗ API Server Error")
            log.error("  Solutions:")
            log.error("    1. Try again later")
            log.error("    2. Contact API administrator")
            log.error("    3. Check API server status page")
            raise APIServerError(f"Server error (500): Internal server error")
            
        elif status_code == 503:
            log.error("✗ API Service Unavailable")
            log.error("  Solutions:")
            log.error("    1. Wait and retry (service may be restarting)")
            log.error("    2. Check API server status page")
            log.error("    3. Contact API administrator")
            raise APIServerError(f"Service unavailable (503): API temporarily unavailable")
            
        else:
            log.error(f"✗ API returned status code: {status_code}")
            log.error(f"  Response: {response.text[:200]}")
            raise APIServerError(f"HTTP {status_code}: {response.text[:200]}")


def build_results_payload(
    system_info: Dict[str, Any],
    test_results: List[Dict[str, Any]],
    execution_time: str,
    test_environment: str = "bare_metal",
    build_info: Optional[Dict[str, Any]] = None,
    deployment_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build complete results payload with system info, test results, and metadata.
    
    Args:
        system_info: System information (OS, CPU, GPU)
        test_results: List of test results
        execution_time: Execution timestamp
        test_environment: Environment type (bare_metal, vm, docker)
        build_info: ROCm build information
        deployment_info: Test deployment information
        
    Returns:
        Complete results payload for API submission
    """
    # Build BM config
    bm_config = {
        # OS Information
        "os_name": system_info.get('os', 'Unknown'),
        "os_version": system_info.get('os_version', 'Unknown'),
        "os_kernel_name": system_info.get('kernel', 'Unknown'),
        "hostname": system_info.get('hostname', 'Unknown'),
        "system_ip": system_info.get('system_ip', '0.0.0.0'),
        
        # CPU Information
        "cpu_model_name": system_info.get('cpu', {}).get('model', 'Unknown'),
        "cpu_cores": system_info.get('cpu', {}).get('cores', 0),
        "cpu_sockets": system_info.get('cpu', {}).get('sockets', 1),
        "cpu_ram_size": format_memory_size(system_info.get('cpu', {}).get('ram_size', 0)),
        "cpu_manufacturer_model": system_info.get('cpu', {}).get('model', 'Unknown'),
        "cpu_numa_nodes": system_info.get('cpu', {}).get('numa_nodes', 1),
        "cpu_clock_speed": format_clock_speed(system_info.get('cpu', {}).get('clock_speed', 0)),
        "cpu_l1_cache": format_cache_size(system_info.get('cpu', {}).get('l1_cache', 0)),
        "cpu_l2_cache": format_cache_size(system_info.get('cpu', {}).get('l2_cache', 0)),
        "cpu_l3_cache": format_cache_size(system_info.get('cpu', {}).get('l3_cache', 0)),
        
        # GPU Information
        "ngpu": system_info.get('gpu', {}).get('count', 0),
        "gpu_name": system_info.get('gpu', {}).get('name', 'Unknown'),
        "gpu_marketing_name": system_info.get('gpu', {}).get('marketing_name', 'Unknown'),
        "gpu_device_id": system_info.get('gpu', {}).get('device_id', None),
        "gpu_revision_id": system_info.get('gpu', {}).get('revision_id', None),
        "gpu_vram_size": format_memory_size(int(system_info.get('gpu', {}).get('vram_size', 0))),
        "gpu_sys_clock": format_clock_speed(system_info.get('gpu', {}).get('sys_clock', 0)),
        "gpu_mem_clock": format_clock_speed(system_info.get('gpu', {}).get('mem_clock', 0)),
        "no_of_nodes": system_info.get('gpu', {}).get('no_of_nodes', 1),
        "xgmi_type": system_info.get('gpu', {}).get('xgmi_type', 'Unknown'),
        "gpu_partition_mode": system_info.get('gpu', {}).get('partition_mode', 'Unknown'),
        "vbios": system_info.get('gpu', {}).get('vbios', 'Unknown'),
        "host_driver": system_info.get('gpu', {}).get('host_driver', 'Unknown'),
        "gpu_firmwares": system_info.get('gpu', {}).get('firmwares', []),
        
        # System BIOS
        "sbios": system_info.get('sbios', 'Unknown'),
    }
    
    # Build test results
    formatted_results = []
    for result in test_results:
        # Get start time or use current time as fallback
        start_time = result.get('start_time', '')
        if not start_time:
            # Generate ISO format timestamp if not provided
            start_time = datetime.now().isoformat()
        
        # Determine test result (PASS/FAIL)
        test_result = "PASS" if result.get('success', False) else "FAIL"
        
        # Build test metrics array from result data
        test_metrics = []
        
        # Check if result has score/metrics data
        if 'score' in result and result['score'] is not None:
            metric = {
                "score": float(result.get('score', 0.0)),
                "unit": result.get('unit', ''),
                "flag": result.get('flag', 'H')  # H (Higher is better) or L (Lower is better)
            }
            # Add optional fields if present
            if 'metric_name' in result:
                metric["metric_name"] = result['metric_name']
            if 'primary' in result:
                metric["primary"] = result['primary']
            
            test_metrics.append(metric)
        
        # Also check for metrics array in result
        if 'metrics' in result and isinstance(result['metrics'], list):
            for m in result['metrics']:
                if isinstance(m, dict) and 'score' in m:
                    metric = {
                        "score": float(m.get('score', 0.0)),
                        "unit": m.get('unit', ''),
                        "flag": m.get('flag', 'H')
                    }
                    if 'metric_name' in m:
                        metric["metric_name"] = m['metric_name']
                    if 'primary' in m:
                        metric["primary"] = m['primary']
                    
                    test_metrics.append(metric)
        
        # API requires at least one metric for PASS results
        # If no metrics found and test passed, add execution time as default metric
        if not test_metrics and test_result == "PASS":
            # Try to get unit and flag from result, otherwise use defaults
            default_unit = result.get('unit', 'seconds')
            default_flag = result.get('flag', 'L')  # Lower is better for execution time
            
            test_metrics.append({
                "metric_name": "execution_time",
                "score": result.get('duration', 0.0),
                "unit": default_unit,
                "flag": default_flag,
                "primary": True
            })
        
        formatted_results.append({
            "test_result": test_result,  # Log parser result (PASS/FAIL)
            "test_start_time": start_time,  # Test start timestamp (ISO format)
            "test_execution_time": result.get('duration', 0.0),
            "test_log": result.get('log_path', ''),  # Log file path
            "test_metrics": test_metrics,  # Metrics from log parser (score, unit, flag)
            "test_config": result.get('test_config', {})  # Test-specific configuration
        })
    
    # Build build_info section
    if build_info is None:
        build_info = {
            "rocm_version": "Unknown",
            "rocm_build_type": "Unknown",
            "rocm_build_lib_type": "Unknown",
            "rocm_package_manager": "Unknown",
            "rocm_package_manager_version": "Unknown",
            "install_type": "Unknown"
        }
    
    # Build deployment_info section
    if deployment_info is None:
        deployment_info = {
            "test_deployed_by": "Unknown",
            "test_deployed_on": datetime.now().isoformat(),
            "execution_label": "",
            "test_flag": "",
            "testcase_command": "",
            "execution_type": "manual"
        }
    
    # Build complete payload
    payload = {
        "test_environment": test_environment,
        "bm_config": bm_config,
        "build_info": build_info,
        "deployment_info": deployment_info,
        "results": formatted_results
    }
    
    return payload


def validate_payload(payload: Dict[str, Any]) -> bool:
    """Validate results payload structure and required fields.
    
    Args:
        payload: Results payload to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required top-level fields
        required_fields = ['test_environment', 'bm_config', 'results']
        for field in required_fields:
            if field not in payload:
                log.error(f"Missing required field: {field}")
                return False
        
        # Check bm_config has required fields
        bm_config = payload['bm_config']
        required_bm_fields = ['os_name', 'cpu_model_name', 'cpu_cores']
        for field in required_bm_fields:
            if field not in bm_config or not bm_config[field]:
                log.error(f"Missing or empty bm_config field: {field}")
                return False
        
        # Check results is a list
        if not isinstance(payload['results'], list):
            log.error("results must be a list")
            return False
        
        # Check each test result has required fields per API schema
        required_result_fields = ['test_result', 'test_start_time', 'test_execution_time', 
                                   'test_log', 'test_metrics', 'test_config']
        for i, result in enumerate(payload['results']):
            for field in required_result_fields:
                if field not in result:
                    log.error(f"Test result {i} missing required field: {field}")
                    return False
            
            # Validate test_result enum
            if result['test_result'] not in ['PASS', 'FAIL']:
                log.error(f"Test result {i} has invalid test_result (must be PASS or FAIL)")
                return False
            
            # Validate test_config has required fields
            test_config = result.get('test_config', {})
            required_config_fields = ['test_name', 'sub_test_name', 'python_version', 'environment_dependencies']
            for field in required_config_fields:
                if field not in test_config:
                    log.error(f"Test result {i} test_config missing required field: {field}")
                    return False
        
        return True
        
    except Exception as e:
        log.error(f"Validation error: {e}")
        return False

