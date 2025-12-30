"""MPI Helper utilities for multi-GPU/multi-node benchmarks.

Provides MPI detection, installation, and setup for benchmarks that require
distributed execution (e.g., RCCL, NCCL).
"""

import subprocess
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MPIHelper:
    """Helper class for MPI setup and management."""

    def __init__(self, install_dir: Optional[Path] = None):
        """Initialize MPI helper.

        Args:
            install_dir: Directory for local MPI installation.
                        If None, uses current directory / "openmpi"
        """
        self.mpi_path: Optional[str] = None
        self.install_dir = install_dir or Path.cwd() / "openmpi"
        self.mpi_version = "4.1.4"

    def setup(self) -> bool:
        """Setup MPI - check for existing installation or install if needed.

        Returns:
            True if MPI is available, False otherwise
        """
        # Check system MPI installations
        for mpi_bin in ["/usr/bin", "/usr/lib64/openmpi/bin", "/opt/openmpi/bin"]:
            if Path(mpi_bin, "mpirun").exists():
                self.mpi_path = mpi_bin
                logger.info(f"Found system MPI at: {self.mpi_path}")
                return True

        # Check local build
        mpirun = self.install_dir / "bin" / "mpirun"
        if mpirun.exists():
            self.mpi_path = str(mpirun.parent)
            logger.info(f"Using local MPI at: {self.mpi_path}")
            return True

        # Build MPI locally
        logger.info("Building OpenMPI locally (this may take several minutes)...")
        if self._build_mpi():
            return True

        logger.warning("MPI unavailable - distributed tests may not run optimally")
        return False

    def _build_mpi(self) -> bool:
        """Download and build OpenMPI locally.

        Returns:
            True if build succeeds, False otherwise
        """
        mpi_url = f"https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-{self.mpi_version}.tar.gz"
        build_dir = self.install_dir.parent
        mpi_src_dir = build_dir / f"openmpi-{self.mpi_version}"

        try:
            # Ensure build directory exists
            build_dir.mkdir(parents=True, exist_ok=True)

            # Download and extract
            tar_file = build_dir / f"openmpi-{self.mpi_version}.tar.gz"
            subprocess.run(
                ["wget", "-q", "-O", str(tar_file), mpi_url], check=True, timeout=300
            )
            subprocess.run(
                ["tar", "-xzf", str(tar_file), "-C", str(build_dir)],
                check=True,
                timeout=60,
            )
            tar_file.unlink()

            # Configure, build, and install
            ncpus = (
                subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip()
                or "4"
            )
            common_opts = {
                "cwd": str(mpi_src_dir),
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.PIPE,
            }

            subprocess.run(
                [
                    "./configure",
                    f"--prefix={self.install_dir}",
                    "--enable-mpi-cxx",
                    "--with-rocm",
                    "--disable-man-pages",
                ],
                check=True,
                timeout=600,
                **common_opts,
            )
            subprocess.run(
                ["make", f"-j{ncpus}"], check=True, timeout=1800, **common_opts
            )
            subprocess.run(["make", "install"], check=True, timeout=300, **common_opts)

            # Cleanup
            subprocess.run(["rm", "-rf", str(mpi_src_dir)], check=False)

            # Verify installation
            mpirun = self.install_dir / "bin" / "mpirun"
            if mpirun.exists():
                self.mpi_path = str(mpirun.parent)
                logger.info(f"OpenMPI built successfully at: {self.mpi_path}")
                return True

            logger.error("MPI build failed - mpirun not found after installation")
            return False

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.error(f"MPI build failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during MPI build: {e}")
            return False

    def get_mpi_path(self) -> Optional[str]:
        """Get the MPI binary path.

        Returns:
            Path to MPI bin directory, or None if not available
        """
        return self.mpi_path

    def get_mpirun_command(self) -> Optional[str]:
        """Get the full path to mpirun binary.

        Returns:
            Path to mpirun, or None if not available
        """
        if self.mpi_path:
            return str(Path(self.mpi_path) / "mpirun")
        return None

    def get_env_vars(self) -> dict:
        """Get environment variables needed for MPI execution.

        Returns:
            Dictionary of environment variables (PATH, LD_LIBRARY_PATH)
        """
        import os

        if not self.mpi_path:
            return {}

        mpi_lib_path = str(Path(self.mpi_path).parent / "lib")
        return {
            "PATH": f"{self.mpi_path}:{os.environ.get('PATH', '')}",
            "LD_LIBRARY_PATH": f"{mpi_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        }
