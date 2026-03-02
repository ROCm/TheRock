"""RDNA2 (gfx103x) GPU family - RX 6000 series."""

from amd_gpu_driver.gpu.family import GPUFamilyConfig, register_gpu_family

_RDNA2_COMMON = dict(
    architecture="RDNA2",
    sh_reg_base=0x2C00,
    sh_reg_end=0x3000,
    uconfig_reg_base=0xC000,
    uconfig_reg_end=0x10000,
    sdma_major=5,
    sdma_minor=2,
    eop_buffer_size=4096,
    ctx_save_restore_size=0x2C02000,
    wave_size=32,
    max_workgroup_size=1024,
)

_RDNA2_TARGETS = [
    ("gfx1030", (10, 3, 0), 2),  # RX 6800/XT
    ("gfx1031", (10, 3, 1), 2),  # RX 6700/XT
    ("gfx1032", (10, 3, 2), 2),  # RX 6600
    ("gfx1033", (10, 3, 3), 1),  # Van Gogh iGPU
    ("gfx1034", (10, 3, 4), 1),  # RX 6500 XT
    ("gfx1035", (10, 3, 5), 1),  # Radeon 680M
    ("gfx1036", (10, 3, 6), 1),  # Raphael iGPU
]

for name, version, num_se in _RDNA2_TARGETS:
    register_gpu_family(
        GPUFamilyConfig(
            name=name,
            gfx_version=version,
            num_shader_engines=num_se,
            **_RDNA2_COMMON,
        )
    )
