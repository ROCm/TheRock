"""Hardware detection for ROCm Test Kit - MI300/MI350 support."""
import logging
import subprocess
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class HardwareInfo:
    """Stores detected hardware information."""

    def __init__(self):
        self.gpus: List[Dict[str, str]] = []
        self.is_mi300_series = False
        self.is_mi350_series = False
        self.gpu_count = 0
        self.compatible = False

    def __str__(self):
        lines = [
            f"GPU Count: {self.gpu_count}",
            f"MI300 Series: {self.is_mi300_series}",
            f"MI350 Series: {self.is_mi350_series}",
            f"Compatible: {self.compatible}",
        ]
        for i, gpu in enumerate(self.gpus):
            lines.append(f"  GPU {i}: {gpu.get('name', 'Unknown')} (gfx{gpu.get('gfx_version', '?')})")
        return "\n".join(lines)


def detect_hardware() -> HardwareInfo:
    """
    Detect available GPUs and determine if they are MI300/MI350 series.

    Returns:
        HardwareInfo object with detected hardware details
    """
    info = HardwareInfo()

    try:
        # Try rocm-smi first
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            info.gpus = _parse_rocm_smi(result.stdout)
        else:
            logger.warning("rocm-smi failed, trying rocminfo...")
            info.gpus = _detect_via_rocminfo()

    except FileNotFoundError:
        logger.warning("rocm-smi not found, trying rocminfo...")
        info.gpus = _detect_via_rocminfo()
    except Exception as e:
        logger.error(f"Error detecting hardware: {e}")
        return info

    # Analyze detected GPUs
    info.gpu_count = len(info.gpus)

    for gpu in info.gpus:
        gfx = gpu.get('gfx_version', '')
        name = gpu.get('name', '').lower()

        # Check for MI300 series (gfx940, gfx941, gfx942)
        if gfx in ['940', '941', '942'] or 'mi300' in name or 'mi3' in name:
            info.is_mi300_series = True

        # Check for MI350 series (gfx950+) - future-proofing
        if gfx.startswith('95') or 'mi350' in name or 'mi35' in name:
            info.is_mi350_series = True

    # Mark as compatible if we have MI300 or MI350 series
    info.compatible = info.is_mi300_series or info.is_mi350_series

    return info


def _parse_rocm_smi(output: str) -> List[Dict[str, str]]:
    """Parse rocm-smi output to extract GPU information."""
    gpus = []
    lines = output.strip().split('\n')

    for line in lines:
        # Look for lines like "GPU[0] : AMD Instinct MI300X"
        match = re.search(r'GPU\[(\d+)\]\s*:\s*(.+)', line)
        if match:
            gpu_id = match.group(1)
            gpu_name = match.group(2).strip()

            # Try to determine gfx version from name
            gfx_version = _infer_gfx_version(gpu_name)

            gpus.append({
                'id': gpu_id,
                'name': gpu_name,
                'gfx_version': gfx_version
            })

    return gpus


def _detect_via_rocminfo() -> List[Dict[str, str]]:
    """Fallback: Use rocminfo to detect GPUs."""
    gpus = []

    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return gpus

        # Parse rocminfo output
        current_gpu = None
        gpu_id = 0

        for line in result.stdout.split('\n'):
            line = line.strip()

            # New agent section
            if line.startswith('Agent ') and 'GPU' in line:
                if current_gpu:
                    gpus.append(current_gpu)
                current_gpu = {'id': str(gpu_id)}
                gpu_id += 1

            # Extract marketing name
            if current_gpu and 'Marketing Name:' in line:
                name = line.split(':', 1)[1].strip()
                current_gpu['name'] = name
                current_gpu['gfx_version'] = _infer_gfx_version(name)

            # Extract gfx version directly
            if current_gpu and 'Name:' in line and 'gfx' in line.lower():
                match = re.search(r'gfx(\d+)', line, re.IGNORECASE)
                if match:
                    current_gpu['gfx_version'] = match.group(1)

        # Add last GPU
        if current_gpu:
            gpus.append(current_gpu)

    except Exception as e:
        logger.error(f"Error running rocminfo: {e}")

    return gpus


def _infer_gfx_version(gpu_name: str) -> str:
    """Infer GFX version from GPU marketing name."""
    name_lower = gpu_name.lower()

    # MI300 series mappings
    if 'mi300' in name_lower or 'mi3' in name_lower:
        if 'mi300x' in name_lower or 'mi3x' in name_lower:
            return '942'  # MI300X typically uses gfx942
        elif 'mi300a' in name_lower or 'mi3a' in name_lower:
            return '940'  # MI300A uses gfx940
        else:
            return '940'  # Default MI300

    # MI350 series (future)
    if 'mi350' in name_lower or 'mi35' in name_lower:
        return '950'  # Assumed for MI350

    # MI250 series
    if 'mi250' in name_lower:
        return '90a'

    # MI210
    if 'mi210' in name_lower:
        return '90a'

    # MI100
    if 'mi100' in name_lower:
        return '908'

    return 'unknown'


def check_compatibility(verbose: bool = False) -> bool:
    """
    Check if current hardware is compatible (MI300/MI350).

    Args:
        verbose: If True, print detailed hardware info

    Returns:
        True if compatible hardware detected
    """
    info = detect_hardware()

    if verbose:
        logger.info("Hardware Detection Results:")
        logger.info(str(info))

    if not info.compatible:
        logger.warning(
            "⚠️  No MI300/MI350 series GPUs detected. "
            "This test kit is optimized for MI300/MI350 hardware."
        )

    return info.compatible


if __name__ == "__main__":
    # Test hardware detection
    logging.basicConfig(level=logging.INFO)
    info = detect_hardware()
    print("=" * 60)
    print("ROCm Hardware Detection")
    print("=" * 60)
    print(info)
    print("=" * 60)

    if info.compatible:
        print("✓ Compatible hardware detected!")
    else:
        print("✗ No MI300/MI350 hardware detected")
