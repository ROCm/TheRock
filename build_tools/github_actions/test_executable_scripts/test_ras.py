#!/usr/bin/env python3
"""
RAS (Reliability, Availability and Serviceability) Test Script

Tests AMD GPU RAS error injection on MI300 family GPUs.
Uses amdgpuras tool for UMC, GFX, SDMA, MMHUB, XGMI PCS and PCIe RAS validation.

Usage:
    sudo python3 test_ras.py --devices 0,1,2,3,4,5,6,7
    sudo python3 test_ras.py --ras-package-url <URL>
"""

import argparse
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAS_PACKAGE_BASE_URL_DEFAULT = "http://10.67.79.109/artifactory/linux-ci-generic-local/amdgpuras-tool/releases/amd-1.6.3/446/"
RAS_PACKAGE_NAME = "amdgpuras"

# RAS Tests - error injection commands
RAS_TESTS = {
    # GFX Block (ID: 2)
    "test_gfx_ce": "amdgpuras -b 2 -s 0 -t 2 -S 30",
    "test_gfx_ue": "amdgpuras -b 2 -s 0 -t 4 -S 30",
    "test_gfx_poison_odecc": "amdgpuras -b 2 -t 8 -S 30",
    "test_gfx_poison_crc": "amdgpuras -b 2 -t 8 -S 30 -m 5",
    # MMHUB Block (ID: 3)
    "test_mmhub_ue": "amdgpuras -b 3 -s 0 -t 4 -S 30",
    # PCIe Block (ID: 5)
    "test_pcie_lcrc_tx_ce": "amdgpuras -b 5 -s 1 -t 2 -m 0 -S 30",
    "test_pcie_lcrc_rx_ce": "amdgpuras -b 5 -s 1 -t 2 -m 1 -S 30",
    "test_pcie_ecrc_tx_ue": "amdgpuras -b 5 -s 1 -t 4 -m 2 -S 30",
    "test_pcie_ecrc_rx_ue": "amdgpuras -b 5 -s 1 -t 4 -m 3 -S 30",
    # XGMI Block (ID: 7)
    "test_xgmi_ce": "amdgpuras -b 7 -s 2 -m 6 -t 2 -S 30",
    "test_xgmi_ue": "amdgpuras -b 7 -s 1 -m 0 -t 4 -S 30",
    # SDMA Block (ID: 1)
    "test_sdma_ue": "amdgpuras -b 1 -s 0 -t 4 -S 30",
    "test_sdma_poison_odecc": "amdgpuras -b 1 -s 130 -t 8 -S 30",
    "test_sdma_poison_crc": "amdgpuras -b 1 -s 130 -t 8 -m 5 -S 30",
    # UMC Block (ID: 0)
    "test_umc_odecc_ce": "amdgpuras -b 0 -s 2 -t 2 -a 0x800 -S 30",
    "test_umc_odecc_ue": "amdgpuras -b 0 -s 2 -t 4 -a 0x800 -S 30",
    "test_umc_crc_read_ue": "amdgpuras -b 0 -s 0 -t 4 -a 0x800 -S 30",
    "test_umc_crc_write_ue": "amdgpuras -b 0 -s 0 -t 4 -m 5 -a 0x800 -S 30",
    "test_umc_parity_single_ue": "amdgpuras -b 0 -s 4 -t 4 -m 1 -S 30",
    "test_umc_parity_persist_ue": "amdgpuras -b 0 -s 4 -t 4 -m 2 -S 30",
}


class RASTestExecutor:
    """RAS Test Executor - runs RAS validation tests on AMD GPUs."""
    
    def __init__(self, ras_package_url=None, devices=None, parallel=True, rocm_path=None):
        self.ras_package_url = ras_package_url or RAS_PACKAGE_BASE_URL_DEFAULT
        self.devices = devices or [0]
        self.parallel = parallel
        self.rocm_path = rocm_path or os.environ.get("ROCM_PATH")
        if not self.rocm_path:
            raise RuntimeError("ROCM_PATH not set. Use --rocm-path or set ROCM_PATH environment variable")
        self.amd_smi = os.path.join(self.rocm_path, "bin", "amd-smi")
        self.os_type = None
        self.package_extension = None
        self.passed_tests = []
        self.failed_tests = []
        self.skipped_tests = []
        # Firmware schema states - detected from amd-smi
        self.single_bit_enabled = False  # CE injection
        self.double_bit_enabled = False  # UE injection

    def run_cmd(self, cmd, privilege=False):
        """Run a command and return the result."""
        if privilege and os.geteuid() != 0:
            cmd = f"sudo {cmd}" if isinstance(cmd, str) else ["sudo"] + cmd
        
        cmd_list = shlex.split(cmd) if isinstance(cmd, str) else cmd
        logger.info(f"++ Exec: {shlex.join(cmd_list)}")
        
        result = subprocess.run(cmd_list, capture_output=True, text=True)
        logger.info(f"Exit code: {result.returncode}")
        if result.stdout:
            logger.info(f"stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"stderr: {result.stderr}")
        return result

    def detect_os_type(self):
        """Detect if OS is RPM-based or DEB-based."""
        if subprocess.run(["which", "dpkg"], capture_output=True).returncode == 0:
            self.os_type, self.package_extension = "deb", ".deb"
            logger.info("Detected DEB-based OS")
        elif subprocess.run(["which", "rpm"], capture_output=True).returncode == 0:
            self.os_type, self.package_extension = "rpm", ".rpm"
            logger.info("Detected RPM-based OS")
        else:
            raise RuntimeError("Could not detect OS type")

    def detect_ras_schemas(self):
        """Detect firmware RAS schema states from amd-smi."""
        logger.info(f"Using amd-smi: {self.amd_smi}")
        result = subprocess.run([self.amd_smi, "static", "-r"], capture_output=True, text=True)
        if result.returncode == 0:
            self.single_bit_enabled = "SINGLE_BIT_SCHEMA: ENABLED" in result.stdout
            self.double_bit_enabled = "DOUBLE_BIT_SCHEMA: ENABLED" in result.stdout
        logger.info(f"Firmware: SINGLE_BIT(CE)={'ENABLED' if self.single_bit_enabled else 'DISABLED'}, "
                    f"DOUBLE_BIT(UE)={'ENABLED' if self.double_bit_enabled else 'DISABLED'}")

    def is_test_supported(self, test_name, cmd):
        """Check if test can run based on firmware schema states."""
        # PCIe uses different mechanism - always run
        if "pcie" in test_name:
            return True, ""
        
        # Extract error type: -t 2 (CE), -t 4 (UE), -t 8 (Poison)
        type_match = re.search(r'-t\s*(\d+)', cmd)
        if not type_match:
            return True, ""
        
        err_type = int(type_match.group(1))
        
        # Check firmware schema states
        if err_type == 2 and not self.single_bit_enabled:
            return False, "SINGLE_BIT_SCHEMA disabled (CE blocked)"
        if err_type == 4 and not self.double_bit_enabled:
            return False, "DOUBLE_BIT_SCHEMA disabled (UE blocked)"
        
        return True, ""

    def install_ras_package(self):
        """Download and install amdgpuras package."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Find package URL
            base_url = self.ras_package_url.rstrip('/') + '/'
            api_url = base_url.replace('/ui/native/', '/artifactory/')
            
            result = self.run_cmd(f"curl -fsSL {api_url}")
            if result.returncode != 0:
                raise RuntimeError(f"Failed to fetch package listing: {result.stderr}")
            
            # Extract filename from href="filename.deb" or href="filename.rpm"
            pattern = rf'href="([^"]+{re.escape(self.package_extension)})"'
            matches = re.findall(pattern, result.stdout, re.IGNORECASE)
            if not matches:
                raise RuntimeError(f"No {self.package_extension} package found")
            
            pkg_filename = matches[0]
            pkg_url = api_url + pkg_filename
            pkg_path = os.path.join(tmp_dir, pkg_filename)
            
            # Download
            logger.info(f"Downloading {pkg_url}...")
            if self.run_cmd(f"curl -fsSL -o {pkg_path} {pkg_url}").returncode != 0:
                raise RuntimeError("Failed to download package")
            
            # Install
            if self.os_type == "rpm":
                self.run_cmd(f"rpm -e {RAS_PACKAGE_NAME}", privilege=True)
                self.run_cmd(f"rpm -ivh --nodeps --force {pkg_path}", privilege=True)
            else:
                self.run_cmd(f"dpkg -r {RAS_PACKAGE_NAME}", privilege=True)
                self.run_cmd(f"dpkg -i --force-all {pkg_path}", privilege=True)
            
            if self.run_cmd("which amdgpuras").returncode != 0:
                raise RuntimeError("amdgpuras not found after installation")
            self.run_cmd("amdgpuras -v", privilege=True)

    def parse_result(self, result, test_name):
        """Parse test result: pass/fail/skip."""
        output = result.stdout + result.stderr
        
        if "Error Inject Successfully" in output or result.returncode == 0:
            logger.info(f"[PASS] {test_name}")
            return "pass"
        if "not support" in output.lower() or "unsupported" in output.lower():
            logger.info(f"[SKIP] {test_name}")
            return "skip"
        
        logger.error(f"[FAIL] {test_name}: {output}")
        return "fail"

    def run_test(self, test_name, cmd, device_id):
        """Run a single test on a device."""
        full_name = f"{test_name}_gpu{device_id}"
        
        # Check if test is supported based on schemas
        supported, reason = self.is_test_supported(test_name, cmd)
        if not supported:
            logger.info(f"[SKIP] {full_name} - {reason}")
            return full_name, "skip"
        
        cmd_with_device = f"{cmd} -d {device_id}" if "-d" not in cmd else re.sub(r'-d\s*\d+', f'-d {device_id}', cmd)
        result = self.run_cmd(cmd_with_device, privilege=True)
        return full_name, self.parse_result(result, full_name)

    def run_tests(self):
        """Run all RAS tests distributed across GPUs."""
        logger.info("=" * 60)
        logger.info("Running RAS Tests")
        logger.info(f"Devices: {self.devices}, Parallel: {self.parallel}")
        logger.info("=" * 60)
        
        # Distribute tests round-robin
        all_tests = [(name, cmd, self.devices[i % len(self.devices)]) 
                     for i, (name, cmd) in enumerate(RAS_TESTS.items())]
        
        if self.parallel and len(self.devices) > 1:
            # Group by device
            by_device = {}
            for name, cmd, dev in all_tests:
                by_device.setdefault(dev, []).append((name, cmd))
            
            with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
                futures = {executor.submit(lambda d, t: [self.run_test(n, c, d) for n, c in t], dev, tests): dev 
                           for dev, tests in by_device.items()}
                for future in as_completed(futures):
                    for name, status in future.result():
                        getattr(self, f"{status}ed_tests" if status != "skip" else "skipped_tests").append(name)
        else:
            for name, cmd, dev in all_tests:
                full_name, status = self.run_test(name, cmd, dev)
                getattr(self, f"{status}ed_tests" if status != "skip" else "skipped_tests").append(full_name)

    def run_eeprom_reset(self):
        """Reset EEPROM on all devices."""
        logger.info("Running EEPROM Reset")
        for dev in self.devices:
            name = f"eeprom_reset_gpu{dev}"
            result = self.run_cmd(f"amdgpuras -C -d {dev}", privilege=True)
            if result.returncode == 0:
                logger.info(f"[PASS] {name}")
                self.passed_tests.append(name)
            else:
                logger.error(f"[FAIL] {name}")
                self.failed_tests.append(name)

    def print_summary(self):
        """Print test summary."""
        logger.info("\n" + "=" * 60)
        logger.info("RAS Test Summary")
        logger.info(f"Devices: {self.devices}")
        logger.info("-" * 60)
        logger.info(f"PASSED:  {len(self.passed_tests)}")
        logger.info(f"FAILED:  {len(self.failed_tests)}")
        logger.info(f"SKIPPED: {len(self.skipped_tests)}")
        if self.failed_tests:
            logger.error(f"Failed: {', '.join(self.failed_tests)}")
        return len(self.failed_tests) == 0

    def execute(self):
        """Execute all tests."""
        logger.info(f"RAS Package URL: {self.ras_package_url}")
        try:
            self.detect_os_type()
            self.detect_ras_schemas()
            self.install_ras_package()
            self.run_tests()
            self.run_eeprom_reset()
            return 0 if self.print_summary() else 1
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return 1


def main():
    parser = argparse.ArgumentParser(description="AMD GPU RAS Test Executor")
    parser.add_argument("--ras-package-url", default=os.environ.get("RAS_PACKAGE_URL", RAS_PACKAGE_BASE_URL_DEFAULT))
    parser.add_argument("--devices", default="0", help="Comma-separated GPU IDs (default: 0)")
    parser.add_argument("--rocm-path", default=os.environ.get("ROCM_PATH"),
                        help="Path to ROCm installation (required: --rocm-path or $ROCM_PATH)")
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--no-parallel", dest="parallel", action="store_false")
    args = parser.parse_args()
    
    devices = [int(d.strip()) for d in args.devices.split(",")]
    executor = RASTestExecutor(args.ras_package_url, devices, args.parallel, args.rocm_path)
    sys.exit(executor.execute())


if __name__ == "__main__":
    main()
