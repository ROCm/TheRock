"""Results handler for building, saving, and uploading test results."""

import sys
import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from decimal import Decimal, InvalidOperation

from ..logger import log
from .results_api import ResultsAPI, build_results_payload, validate_payload
from ..constants import Constants, SEPARATOR_LINE


class ResultsHandler:
    """Static methods for building, saving, and uploading test results."""

    @staticmethod
    def build_deployment_info(
        config: Optional[Any] = None,
        deployed_by: str = "",
        execution_label: str = "",
        ci_group: str = "therock_pr",
    ) -> Dict[str, str]:
        """Build deployment information dict for test execution.

        Args:
            config: Configuration object (optional)
            deployed_by: Username who ran the test
            execution_label: Execution label
            ci_group: CI/CD group identifier (default: therock_pr)

        Returns:
            Dict: Deployment info with timestamp, user, label, command, and CI group
        """

        # Build deployment info
        deployment_info = {
            "test_deployed_by": deployed_by,
            "test_deployed_on": datetime.now().isoformat(),
            "execution_label": execution_label,
            "test_flag": "prod_test",
            "testcase_command": " ".join(sys.argv),
            "execution_type": "automated",
            "ci_group": ci_group,
        }

        log.debug(
            f"Deployment info: deployed_by={deployed_by}, ci_group={ci_group}, command={deployment_info['testcase_command']}"
        )

        return deployment_info

    @staticmethod
    def build_system_info_dict(system_context: Any) -> Dict[str, Any]:
        """Build system info dict from SystemContext for API payload.

        Args:
            system_context: SystemContext object with detected system info

        Returns:
            Dict: System information (OS, CPU, GPU details)
        """
        return {
            "os": f"{system_context.os_name} {system_context.os_version}",
            "os_version": system_context.os_version,
            "kernel": system_context.kernel,
            "hostname": system_context.hostname,
            "system_ip": system_context.system_ip,
            "cpu": {
                "model": system_context.cpu_model,
                "cores": system_context.cpu_cores,
                "sockets": system_context.cpu_sockets,
                "ram_size": system_context.cpu_ram_size,
                "numa_nodes": system_context.cpu_numa_nodes,
                "clock_speed": system_context.cpu_clock_speed,
                "l1_cache": system_context.cpu_l1_cache,
                "l2_cache": system_context.cpu_l2_cache,
                "l3_cache": system_context.cpu_l3_cache,
            },
            "gpu": {
                "count": system_context.gpu_count,
                "name": system_context.gpu_name,
                "marketing_name": system_context.gpu_marketing_name,
                "device_id": system_context.gpu_device_id,
                "revision_id": system_context.gpu_revision_id,
                "vram_size": system_context.gpu_vram_size,
                "sys_clock": system_context.gpu_sys_clock,
                "mem_clock": system_context.gpu_mem_clock,
                "vbios": system_context.gpu_vbios,
                "partition_mode": system_context.gpu_partition_mode,
                "xgmi_type": system_context.gpu_xgmi_type,
                "host_driver": system_context.gpu_host_driver,
                "firmwares": system_context.gpu_firmwares,
                "no_of_nodes": system_context.gpu_count,
                "devices": system_context.gpu_devices,
            },
        }

    @staticmethod
    def save_local_results(
        results_data: Dict[str, Any], output_dir: str, timestamp: Optional[str] = None
    ) -> Optional[Path]:
        """Save results to local JSON file with timestamp.

        Args:
            results_data: Results data dict
            output_dir: Output directory path
            timestamp: Optional timestamp (auto-generated if None)

        Returns:
            Path: Saved file path or None if failed
        """
        try:
            results_dir = Path(output_dir)
            results_dir.mkdir(parents=True, exist_ok=True)

            if timestamp is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")

            output_file = results_dir / f"results_{timestamp}.json"
            with open(output_file, "w") as f:
                json.dump(results_data, f, indent=2)

            log.info(f"Results saved: {output_file}")
            return output_file
        except Exception as e:
            log.error(f"Failed to save local results: {e}")
            return None

    @staticmethod
    def upload_to_api(
        system_info: Dict[str, Any],
        test_results: List[Dict[str, Any]],
        timestamp: str,
        api_config: Dict[str, Any],
        rocm_info: Dict[str, Any],
        deployment_info: Dict[str, str],
        test_environment: str = Constants.TEST_ENV_BARE_METAL,
    ) -> bool:
        """Upload test results to API with system context.

        Args:
            system_info: System info dict
            test_results: List of test result dicts
            timestamp: Execution timestamp
            api_config: API config from ConfigHelper
            rocm_info: ROCm info dict
            deployment_info: Deployment info dict
            test_environment: Environment type (bm/vm/docker)

        Returns:
            bool: True if successful, False otherwise
        """
        # Check if API submission is enabled
        if not api_config.get("enabled", False):
            log.debug("API submission disabled")
            return False

        api_url = api_config.get("url", "")
        fallback_url = api_config.get("fallback_url", "")
        api_key = api_config.get("api_key", "")

        if not api_url:
            log.warning("API URL not configured, skipping submission")
            return False

        log.info(SEPARATOR_LINE)
        log.info("Submitting results to API...")
        log.info(SEPARATOR_LINE)

        try:
            # Build API payload
            test_results_for_api = []
            for result in test_results:
                test_results_for_api.append(
                    {
                        "test_name": result.get("test_name", ""),
                        "success": result.get("status", "PASS") == "PASS"
                        or result.get("success", False),
                        "duration": result.get("duration", 0),
                        "error_message": result.get("error_message", ""),
                        "score": result.get("score"),  # Test score/value
                        "unit": result.get("unit", ""),  # Unit of measurement
                        "flag": result.get(
                            "flag", "H"
                        ),  # H (Higher is better) or L (Lower is better)
                        "test_config": result.get(
                            "test_config", {}
                        ),  # Test-specific configuration
                        "start_time": result.get(
                            "start_time", ""
                        ),  # Test start timestamp
                        "log_path": result.get("log_path", ""),  # Log file path
                    }
                )

            payload = build_results_payload(
                system_info=system_info,
                test_results=test_results_for_api,
                execution_time=timestamp,
                test_environment=test_environment,
                build_info=rocm_info,
                deployment_info=deployment_info,
            )

            log.debug(payload)

            # Validate payload
            if not validate_payload(payload):
                log.error("Payload validation failed, skipping API submission")
                return False

            # Submit to API
            api_client = ResultsAPI(api_url, api_key, fallback_url)
            success = api_client.submit_results(payload)

            if success:
                log.info("Results submitted to API successfully")
                return True
            else:
                log.warning("Failed to submit results to API")
                return False

        except Exception as e:
            log.error(f"Unexpected error submitting to API: {e}")
            log.warning("Results not submitted - unexpected error")
            return False

    @staticmethod
    def fetch_lkg_scores_from_api(
        test_name: str, api_config: Dict[str, Any], rocm_info: Dict[str, Any]
    ) -> Dict[Tuple[str, str], float]:
        """Fetch Last Known Good (LKG) scores from API for comparison.

        Args:
            test_name: Test name for LKG score lookup
            api_config: API config from ConfigHelper
            rocm_info: ROCm info dict

        Returns:
            Dict: Mapping of (test_name, sub_test_name) tuples to LKG scores
        """
        try:
            # Get ROCm version
            rocm_version = rocm_info.get("rocm_version", "")

            # Validate API URL
            api_url = api_config.get("url", "")
            if not api_url:
                raise ValueError("API URL not configured in api_config.")

            # Build query string
            fetch_api_url = f"{api_url}/api/v1/rock-ci-results?skip=0&limit=600&rocm_version={rocm_version}&{test_name}&lkg_score=true"

            # Make API request
            response = requests.get(
                fetch_api_url, headers={"accept": "application/json"}
            )
            response.raise_for_status()  # Raise error for HTTP issues
            data = response.json()

            # Validate response structure
            if "results" not in data:
                raise ValueError("Invalid API response: 'results' key not found.")

            # Build dictionary of LKG scores
            lkg_scores = {}
            for result in data.get("results", []):
                t_name = result.get("test_config", {}).get("test_name")
                sub_test_name = result.get("test_config", {}).get("sub_test_name")
                for metric in result.get("test_metrics", []):
                    lkg_score = metric.get("lkg_score")
                    if t_name and sub_test_name and lkg_score is not None:
                        lkg_scores[(t_name, sub_test_name)] = float(lkg_score)

            return lkg_scores

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
        except ValueError as e:
            raise ValueError(f"Data error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")

    @staticmethod
    def compare_results_with_lkg(
        test_results: List[Dict[str, Any]], lkg_scores: Dict[Tuple[str, str], float]
    ) -> List[Dict[str, Any]]:
        """Compare test results with LKG baseline and add comparison data.

        Args:
            test_results: List of test result dicts
            lkg_scores: Mapping of (test_name, subtest) to LKG scores

        Returns:
            List[Dict[str, Any]]: Test results with added lkg_score, diff_pct, final_result fields
        """
        compared_results = []

        for result in test_results:
            # Create a copy to avoid modifying original
            compared = result.copy()

            test_name = result.get("test_name", "")
            subtest = result.get("subtest", "")
            score = result.get("score", 0.0)
            flag = result.get("flag", "H")

            # Get LKG score
            lkg_score = lkg_scores.get((test_name, subtest))
            diff: Optional[Decimal] = None
            final_result = "UNKNOWN"

            if lkg_score is not None:
                try:
                    score_dec = Decimal(str(score))
                    lkg_score_dec = Decimal(str(lkg_score))

                    if flag == "H" and lkg_score_dec != 0:
                        diff = ((score_dec - lkg_score_dec) / lkg_score_dec) * 100
                    elif flag == "L" and lkg_score_dec != 0:
                        diff = ((lkg_score_dec - score_dec) / lkg_score_dec) * 100

                    # Determine FinalResult
                    if diff is not None:
                        final_result = "FAIL" if diff < -5 else "PASS"

                except (InvalidOperation, ValueError) as e:
                    log.warning(
                        f"Error calculating diff for {test_name}/{subtest}: {e}"
                    )

            # Add LKG comparison fields to result
            compared["lkg_score"] = float(lkg_score) if lkg_score is not None else None
            compared["diff_pct"] = round(float(diff), 2) if diff is not None else None
            compared["final_result"] = final_result

            compared_results.append(compared)

        return compared_results
