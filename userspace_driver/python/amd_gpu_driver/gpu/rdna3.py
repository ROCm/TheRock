"""RDNA3 (gfx110x) GPU family - RX 7000 series."""

from amd_gpu_driver.gpu.family import GPUFamilyConfig, register_gpu_family

_RDNA3_COMMON = dict(
    architecture="RDNA3",
    sh_reg_base=0x2C00,
    sh_reg_end=0x3000,
    uconfig_reg_base=0xC000,
    uconfig_reg_end=0x10000,
    sdma_major=6,
    sdma_minor=0,
    eop_buffer_size=4096,
    ctx_save_restore_size=0x2C02000,
    wave_size=32,
    max_workgroup_size=1024,
)

_RDNA3_TARGETS = [
    ("gfx1100", (11, 0, 0), 3),  # RX 7900 XTX
    ("gfx1101", (11, 0, 1), 2),  # RX 7800 XT
    ("gfx1102", (11, 0, 2), 1),  # RX 7700S
    ("gfx1103", (11, 0, 3), 1),  # Radeon 780M
]

for name, version, num_se in _RDNA3_TARGETS:
    register_gpu_family(
        GPUFamilyConfig(
            name=name,
            gfx_version=version,
            num_shader_engines=num_se,
            **_RDNA3_COMMON,
        )
    )
