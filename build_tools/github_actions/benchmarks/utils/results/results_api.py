"""API client for submitting benchmark test results to the configured results tracking API.

Uploads performance metrics, system metadata, and test configurations for regression detection and analysis.
"""

import json
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from jsonschema import validate, ValidationError

from ..logger import log
from ..system import format_memory_size, format_cache_size, format_clock_speed


class ResultsAPI:
    """API client for submitting test results with authentication and fallback support."""

    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        fallback_url: Optional[str] = None,
    ):
        """Initialize API client.

        Args:
            api_url: Base URL for the primary API
            api_key: Optional API key for authentication
            fallback_url: Optional fallback URL if primary fails
        """
        self.api_url = api_url.rstrip("/")
        self.fallback_url = fallback_url.rstrip("/") if fallback_url else None
        self.api_key = api_key
        self.session = requests.Session()

        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )

    def submit_results(self, payload: Dict[str, Any]) -> bool:
        """Submit test results to API with fallback support.

        Args:
            payload: Results payload dictionary

        Returns:
            True if successful, False otherwise
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

    def _try_submit(
        self, base_url: str, payload: Dict[str, Any], is_fallback: bool = False
    ) -> bool:
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

            response = self.session.post(endpoint, json=payload, timeout=30)

            # Raise HTTPError for bad status codes (4xx, 5xx)
            response.raise_for_status()

            # If we reach here, request was successful
            log.info(f"✓ Results submitted successfully to {url_type} API")
            try:
                response_data = response.json()
                log.debug(f"API Response: {json.dumps(response_data, indent=2)}")
            except Exception:
                log.debug(f"Response text: {response.text[:500]}")
            return True

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
            status_code = e.response.status_code if e.response else "Unknown"
            error_msg = (
                e.response.text[:200] if e.response and e.response.text else str(e)
            )
            log.warning(
                f"✗ {url_type.capitalize()} API Error ({status_code}): {error_msg}"
            )
            return False

        except json.JSONDecodeError as e:
            url_type = "fallback" if is_fallback else "primary"
            log.warning(f"✗ {url_type.capitalize()} API Invalid JSON Response: {e}")
            return False

        except Exception as e:
            url_type = "fallback" if is_fallback else "primary"
            log.warning(f"✗ {url_type.capitalize()} API Unexpected Error: {e}")
            return False


def build_results_payload(
    system_info: Dict[str, Any],
    test_results: List[Dict[str, Any]],
    execution_time: str,
    test_environment: str = "bare_metal",
    build_info: Optional[Dict[str, Any]] = None,
    deployment_info: Optional[Dict[str, Any]] = None,
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
    # Build Baremetal config
    bm_config = {
        # OS Information
        "os_name": system_info.get("os", "Unknown"),
        "os_version": system_info.get("os_version", "Unknown"),
        "os_kernel_name": system_info.get("kernel", "Unknown"),
        "hostname": system_info.get("hostname", "Unknown"),
        "system_ip": system_info.get("system_ip", "0.0.0.0"),
        # CPU Information
        "cpu_model_name": system_info.get("cpu", {}).get("model", "Unknown"),
        "cpu_cores": system_info.get("cpu", {}).get("cores", 0),
        "cpu_sockets": system_info.get("cpu", {}).get("sockets", 1),
        "cpu_ram_size": format_memory_size(
            system_info.get("cpu", {}).get("ram_size", 0)
        ),
        "cpu_manufacturer_model": system_info.get("cpu", {}).get("model", "Unknown"),
        "cpu_numa_nodes": system_info.get("cpu", {}).get("numa_nodes", 1),
        "cpu_clock_speed": format_clock_speed(
            system_info.get("cpu", {}).get("clock_speed", 0)
        ),
        "cpu_l1_cache": format_cache_size(
            system_info.get("cpu", {}).get("l1_cache", 0)
        ),
        "cpu_l2_cache": format_cache_size(
            system_info.get("cpu", {}).get("l2_cache", 0)
        ),
        "cpu_l3_cache": format_cache_size(
            system_info.get("cpu", {}).get("l3_cache", 0)
        ),
        # GPU Information
        "ngpu": system_info.get("gpu", {}).get("count", 0),
        "gpu_name": system_info.get("gpu", {}).get("name", "Unknown"),
        "gpu_marketing_name": system_info.get("gpu", {}).get(
            "marketing_name", "Unknown"
        ),
        "gpu_device_id": system_info.get("gpu", {}).get("device_id", None),
        "gpu_revision_id": system_info.get("gpu", {}).get("revision_id", None),
        "gpu_vram_size": format_memory_size(
            int(system_info.get("gpu", {}).get("vram_size", 0))
        ),
        "gpu_sys_clock": format_clock_speed(
            system_info.get("gpu", {}).get("sys_clock", 0)
        ),
        "gpu_mem_clock": format_clock_speed(
            system_info.get("gpu", {}).get("mem_clock", 0)
        ),
        "no_of_nodes": system_info.get("gpu", {}).get("no_of_nodes", 1),
        "xgmi_type": system_info.get("gpu", {}).get("xgmi_type", "Unknown"),
        "gpu_partition_mode": system_info.get("gpu", {}).get(
            "partition_mode", "Unknown"
        ),
        "vbios": system_info.get("gpu", {}).get("vbios", "Unknown"),
        "host_driver": system_info.get("gpu", {}).get("host_driver", "Unknown"),
        "gpu_firmwares": system_info.get("gpu", {}).get("firmwares", []),
        # System BIOS
        "sbios": system_info.get("sbios", "Unknown"),
    }

    # Build test results
    formatted_results = []
    for result in test_results:
        # Get start time or use current time as fallback
        start_time = result.get("start_time", "")
        if not start_time:
            # Generate ISO format timestamp if not provided
            start_time = datetime.now().isoformat()

        # Determine test result (PASS/FAIL)
        test_result = "PASS" if result.get("success", False) else "FAIL"

        # Build test metrics array from result data
        test_metrics = []

        # Check if result has score/metrics data
        if "score" in result and result["score"] is not None:
            metric = {
                "score": float(result.get("score", 0.0)),
                "unit": result.get("unit", ""),
                "flag": result.get(
                    "flag", "H"
                ),  # H (Higher is better) or L (Lower is better)
            }
            # Add optional fields if present
            if "metric_name" in result:
                metric["metric_name"] = result["metric_name"]
            if "primary" in result:
                metric["primary"] = result["primary"]

            test_metrics.append(metric)

        # Also check for metrics array in result
        if "metrics" in result and isinstance(result["metrics"], list):
            for m in result["metrics"]:
                if isinstance(m, dict) and "score" in m:
                    metric = {
                        "score": float(m.get("score", 0.0)),
                        "unit": m.get("unit", ""),
                        "flag": m.get("flag", "H"),
                    }
                    if "metric_name" in m:
                        metric["metric_name"] = m["metric_name"]
                    if "primary" in m:
                        metric["primary"] = m["primary"]

                    test_metrics.append(metric)

        # API requires at least one metric for PASS results
        # If no metrics found and test passed, add execution time as default metric
        if not test_metrics and test_result == "PASS":
            # Try to get unit and flag from result, otherwise use defaults
            default_unit = result.get("unit", "seconds")
            default_flag = result.get("flag", "L")  # Lower is better for execution time

            test_metrics.append(
                {
                    "metric_name": "execution_time",
                    "score": result.get("duration", 0.0),
                    "unit": default_unit,
                    "flag": default_flag,
                    "primary": True,
                }
            )

        formatted_results.append(
            {
                "test_result": test_result,  # Log parser result (PASS/FAIL)
                "test_start_time": start_time,  # Test start timestamp (ISO format)
                "test_execution_time": result.get("duration", 0.0),
                "test_log": result.get("log_path", ""),  # Log file path
                "test_metrics": test_metrics,  # Metrics from log parser (score, unit, flag)
                "test_config": result.get(
                    "test_config", {}
                ),  # Test-specific configuration
            }
        )

    # Build build_info section
    if build_info is None:
        build_info = {
            "rocm_version": "Unknown",
            "rocm_build_type": "Unknown",
            "rocm_build_lib_type": "Unknown",
            "rocm_package_manager": "Unknown",
            "rocm_package_manager_version": "Unknown",
            "install_type": "Unknown",
        }

    # Build deployment_info section
    if deployment_info is None:
        deployment_info = {
            "test_deployed_by": "Unknown",
            "test_deployed_on": datetime.now().isoformat(),
            "execution_label": "",
            "test_flag": "",
            "testcase_command": "",
            "execution_type": "manual",
        }

    # Build complete payload
    payload = {
        "test_environment": test_environment,
        "bm_config": bm_config,
        "build_info": build_info,
        "deployment_info": deployment_info,
        "results": formatted_results,
    }

    return payload


def _load_schema() -> Dict[str, Any]:
    """Load JSON schema from file.

    Returns:
        Dict containing the JSON schema
    """
    schema_path = Path(__file__).parent / "schemas" / "payload_schema.json"
    with open(schema_path, "r") as f:
        return json.load(f)


def validate_payload(payload: Dict[str, Any]) -> bool:
    """Validate results payload structure and required fields using JSON Schema.

    Args:
        payload: Results payload to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        schema = _load_schema()
        validate(instance=payload, schema=schema)
        return True
    except ValidationError as e:
        log.error(f"Validation error: {e.message}")
        log.error(f"  Failed at: {' -> '.join(str(p) for p in e.path)}")
        return False
    except Exception as e:
        log.error(f"Unexpected validation error: {e}")
        return False
