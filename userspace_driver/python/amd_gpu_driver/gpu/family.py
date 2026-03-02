"""GPU family configuration dataclass and registry."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GPUFamilyConfig:
    """Configuration for a specific GPU family/architecture."""

    name: str  # e.g. "gfx942"
    architecture: str  # e.g. "CDNA3"
    gfx_version: tuple[int, int, int]  # (major, minor, stepping)

    # Shader register ranges
    sh_reg_base: int = 0x2C00
    sh_reg_end: int = 0x3000

    # UConfig register ranges
    uconfig_reg_base: int = 0xC000
    uconfig_reg_end: int = 0x10000

    # SDMA version
    sdma_major: int = 0
    sdma_minor: int = 0

    # Buffer sizes
    eop_buffer_size: int = 4096
    ctx_save_restore_size: int = 0x2C02000  # ~44MB default

    # Wave configuration
    wave_size: int = 64  # 32 for RDNA, 64 for CDNA/GCN

    # Number of shader engines
    num_shader_engines: int = 1

    # Maximum workgroup size
    max_workgroup_size: int = 1024

    @property
    def gfx_target_version(self) -> int:
        """Encode gfx_version as the kernel's decimal-packed format.

        The Linux KFD topology reports gfx_target_version as a decimal
        encoding: major * 10000 + minor * 100 + stepping.
        E.g. gfx942 = 90402, gfx1100 = 110000, gfx90a = 90010.
        """
        major, minor, stepping = self.gfx_version
        return major * 10000 + minor * 100 + stepping


# Registry mapping gfx_target_version -> GPUFamilyConfig
_registry: dict[int, GPUFamilyConfig] = {}


def register_gpu_family(config: GPUFamilyConfig) -> None:
    """Register a GPU family configuration."""
    _registry[config.gfx_target_version] = config


def get_gpu_family(gfx_target_version: int) -> GPUFamilyConfig | None:
    """Look up GPU family config by gfx_target_version."""
    return _registry.get(gfx_target_version)


def get_all_families() -> dict[int, GPUFamilyConfig]:
    """Return all registered GPU family configs."""
    return dict(_registry)
