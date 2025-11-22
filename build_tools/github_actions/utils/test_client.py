"""Test Client for system detection and result reporting.

Unified interface for collecting system information (OS, hardware, ROCm) and
uploading test/benchmark results to API or local storage.
"""

import time
from prettytable import PrettyTable
from typing import Dict, List, Optional, Any

# Import framework components
from .logger import log
from .constants import Constants
from .exceptions import ConfigurationError

# Import shared utilities
from .system import SystemContext, SystemDetector
from .config import ConfigHelper
from .results import ResultsHandler


class TestClient:
    """Client for system detection, result collection, and API upload.
    
    Attributes:
        config: Configuration dictionary
        config_file: Path to configuration file
        system_detector: SystemDetector instance
        system_context: SystemContext with detected info (or None if not detected)
    
    Example:
        >>> client = TestClient()
        >>> results = [{"test_name": "fft_1024", "score": 1234.5, "unit": "GFLOPS"}]
        >>> client.upload_results("rocfft_benchmark", results)
    """
    
    def __init__(self, config_file: Optional[str] = None, auto_detect: bool = True):
        """Initialize test client.
        
        Args:
            config_file: Path to config file (searches standard locations if None)
            auto_detect: Auto-detect system info on init (default: True)
        """
        # Use ConfigHelper for configuration management
        self.config_file = ConfigHelper.find_config_file(config_file)
        self.config = ConfigHelper.load_config(self.config_file, required=False)
        
        # Configure logging
        ConfigHelper.configure_logging(self.config)
        
        # Initialize system detector
        self.system_detector = SystemDetector()
        self.system_context = None
        
        # Auto-detect system if requested
        if auto_detect:
            self.detect_system()
    
    def detect_system(self):
        """Detect and cache system information.
        
        Collects: Platform (OS, kernel), Hardware (CPU, GPU, memory),
        ROCm (version, build), and BIOS info. Stores in self.system_context.
        """
        # Use SystemDetector for detection
        self.system_context = self.system_detector.detect_all(verbose=True)
    
    def get_system_context(self) -> SystemContext:
        """Get cached system context.
        
        Returns:
            SystemContext: Detected system information
            
        Raises:
            RuntimeError: If system detection hasn't been run
        """
        if self.system_context is None:
            raise RuntimeError(
                "System context not available. "
                "Call detect_system() first or initialize with auto_detect=True"
            )
        return self.system_context
    
    def upload_results(self,
                      test_name: str,
                      test_results: List[Dict[str, Any]],
                      test_status: str = "PASS",
                      test_metadata: Optional[Dict[str, Any]] = None,
                      save_local: bool = True,
                      output_dir: str = "./results") -> bool:
        """Upload test results with system context to API and/or local storage.
        
        Args:
            test_name: Test identifier (e.g., "rocfft_benchmark")
            test_results: List of dicts with test_name, score/value, unit, status
            test_status: Overall status - "PASS", "FAIL", or "SKIP"
            test_metadata: Additional test metadata
            save_local: Save to timestamped JSON file
            output_dir: Directory for local results
        
        Returns:
            bool: True if upload succeeded, False otherwise
        """
        # Ensure system context is available
        if self.system_context is None:
            log.warning("System context not available, detecting now...")
            self.detect_system()
        
        # Build results data using ResultsHandler
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        system_info = ResultsHandler.build_system_info_dict(self.system_context)
        
        results_data = {
            'execution_time': timestamp,
            'test_name': test_name,
            'test_status': test_status,
            'total_tests': len(test_results),
            'passed': sum(1 for r in test_results if r.get('status', 'PASS') == 'PASS'),
            'failed': sum(1 for r in test_results if r.get('status', 'PASS') == 'FAIL'),
            'system_info': system_info,
            'sbios': self.system_context.sbios,
            'rocm_info': self.system_detector.rocm_info,
            'test_results': test_results,
            'test_metadata': test_metadata or {}
        }
        
        # Save locally if requested
        if save_local:
            ResultsHandler.save_local_results(results_data, output_dir, timestamp)
        
        # Upload to API if configured
        return self._upload_to_api(system_info, test_results, timestamp)
    
    def _upload_to_api(self, system_info: Dict, test_results: List[Dict], timestamp: str) -> bool:
        """Upload results to API endpoint.
        
        Args:
            system_info: System context dict
            test_results: List of test result dicts
            timestamp: Execution timestamp (YYYYMMDD_HHMMSS)
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Get API configuration
        api_config = ConfigHelper.get_api_config(self.config)
        
        # Build deployment info
        deployed_by = ConfigHelper.get_deployed_user(self.config)
        execution_label = ConfigHelper.get_execution_label(self.config)
        ci_group = ConfigHelper.get_ci_group(self.config)
        deployment_info = ResultsHandler.build_deployment_info(self.config, deployed_by, execution_label, ci_group)
        
        # Use ResultsHandler for upload
        return ResultsHandler.upload_to_api(
            system_info=system_info,
            test_results=test_results,
            timestamp=timestamp,
            api_config=api_config,
            rocm_info=self.system_detector.rocm_info,
            deployment_info=deployment_info,
            test_environment=Constants.TEST_ENV_BARE_METAL
        )
    
    def print_system_summary(self):
        """Print detected system information to console."""
        if self.system_context is None:
            log.error("System context not available. Run detect_system() first.")
            return
        
        # Use SystemDetector for printing
        self.system_detector.print_system_summary(self.system_context)
    
    def compare_results(self,
                        test_name: str,
                        table: PrettyTable) -> PrettyTable:
        """Compare test results against Last Known Good (LKG) scores from API.

        Args:
            test_name: Test identifier for LKG lookup
            table: PrettyTable with test results

        Returns:
            PrettyTable: Table enriched with LKG comparison columns
        """
        # Get API configuration
        api_config = ConfigHelper.get_api_config(self.config)

        # Fetch LKG scores info
        lkg_scores = ResultsHandler.fetch_lkg_scores_from_api(test_name, api_config, self.system_detector.rocm_info)

        # Compute final results data using ResultsHandler
        return ResultsHandler.get_final_result_table(
            table=table,
            lkg_scores=lkg_scores
        )
